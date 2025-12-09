#!/usr/bin/env python3
"""
PDF Watermark Remover - Clean Implementation
============================================
A modular watermark detection and removal tool with GUI support.

GUI Options:
- PyQt6 (preferred, if available)
- tkinter (fallback, always available with Python)

Only mandatory dependency: PyMuPDF (fitz)

Detection Strategy:
- Watermarks are defined as elements that repeat across most/all pages
- Each detection case handles a specific watermark type
- Cases can be combined for comprehensive detection

Supported Cases:
- Case #1: Diagonal text in main content stream (DiagonalTextDetector)
- Case #2: XObject Form-based watermarks with internal rotation (XObjectWatermarkDetector)
- Case #3: XObject Form-based watermarks with external rotation (XObjectWatermarkDetector)
- Case #4: Deeply nested XObject watermarks in page content streams (NestedXObjectWatermarkDetector)
- Case #5: Horizontal text watermarks like CONFIDENTIAL, DO NOT DISTRIBUTE (HorizontalTextWatermarkDetector)
- Case #6: Vector/path-based watermarks - text drawn as filled paths (VectorWatermarkDetector)
"""

import sys
import os
import math
import re
import subprocess
import platform
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
from functools import lru_cache

# Mandatory dependency
import fitz  # PyMuPDF

# Try to import PyQt6, fall back to tkinter
USING_PYQT6 = False
USING_TKINTER = False

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QProgressBar, QTextEdit, QSplitter,
        QScrollArea, QMessageBox, QFileDialog, QGroupBox, QCheckBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QFont, QIcon
    USING_PYQT6 = True
except ImportError:
    # Fall back to tkinter
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    USING_TKINTER = True
    
    # Try to import PIL for image preview in tkinter
    try:
        from PIL import Image, ImageTk
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False


# =============================================================================
# PRECOMPILED REGEX PATTERNS (module-level for performance)
# =============================================================================

# Matrix patterns for detecting transforms
RE_CM_MATRIX = re.compile(
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm"
)
RE_TM_MATRIX = re.compile(
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm"
)

# Text block pattern
RE_BT_ET = re.compile(r"BT\s.*?\sET", re.DOTALL)

# XObject invocation pattern
RE_DO_CALL = re.compile(r"/(\w+)\s+Do")

# BBox extraction pattern
RE_BBOX = re.compile(r'/BBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]')

# Quick check for diagonal values (~0.707 for 45°)
RE_DIAGONAL_QUICK = re.compile(r'[-]?0\.7\d*\s+[-]?0\.7\d*')

# Precompute common trig values
_DEG_TO_RAD = math.pi / 180
_RAD_TO_DEG = 180 / math.pi


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1024)
def _compute_angle(a: float, b: float) -> float:
    """Compute angle from matrix coefficients with caching."""
    angle = abs(math.atan2(b, a) * _RAD_TO_DEG)
    if angle > 90:
        angle = 180 - angle
    return angle


def is_diagonal_angle(a: float, b: float, min_angle: float = 15.0, max_angle: float = 75.0) -> bool:
    """Fast check if matrix represents diagonal rotation."""
    try:
        angle = _compute_angle(a, b)
        return min_angle <= angle <= max_angle
    except (ValueError, TypeError):
        return False


def open_file_cross_platform(path: str):
    """Open a file with the system's default application."""
    try:
        if not path or not os.path.exists(path):
            return
        system = platform.system()
        if system == "Darwin":
            subprocess.call(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


def normalize_position(bbox: Tuple[float, float, float, float],
                       page_rect: fitz.Rect) -> Tuple[float, float]:
    """Normalize bbox center to 0-1 range relative to page."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    norm_x = (cx - page_rect.x0) / page_rect.width
    norm_y = (cy - page_rect.y0) / page_rect.height
    return (norm_x, norm_y)


def calculate_area_fraction(bbox: Tuple[float, float, float, float],
                            page_rect: fitz.Rect) -> float:
    """Calculate what fraction of the page this bbox covers."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    bbox_area = width * height
    page_area = page_rect.width * page_rect.height
    return bbox_area / page_area if page_area > 0 else 0


def direction_to_angle(direction: Tuple[float, float]) -> float:
    """Convert direction vector to angle in degrees."""
    if not direction or len(direction) < 2:
        return 0.0
    dx, dy = direction[0], direction[1]
    return math.degrees(math.atan2(dy, dx))


def is_diagonal(angle: float, tolerance: float = 10.0) -> bool:
    """Check if angle represents diagonal text (not horizontal/vertical)."""
    abs_angle = abs(angle)
    horizontal_angles = [0, 180, 360]
    vertical_angles = [90, 270]

    for h in horizontal_angles:
        if abs(abs_angle - h) < tolerance:
            return False
    for v in vertical_angles:
        if abs(abs_angle - v) < tolerance:
            return False
    return True


def position_bin(norm_x: float, norm_y: float, granularity: int = 10) -> Tuple[int, int]:
    """Convert normalized position to a discrete bin for comparison."""
    return (int(norm_x * granularity), int(norm_y * granularity))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WatermarkCandidate:
    """Represents a potential watermark element found on a page."""
    page_num: int
    bbox: Tuple[float, float, float, float]
    element_type: str

    norm_position: Tuple[float, float] = (0.0, 0.0)
    area_fraction: float = 0.0

    text_content: str = ""
    font_name: str = ""
    font_size: float = 0.0
    direction_angle: float = 0.0
    color: int = 0

    signature: str = ""

    def __post_init__(self):
        if not self.signature:
            self.signature = self._generate_signature()

    def _generate_signature(self) -> str:
        pos_bin = position_bin(self.norm_position[0], self.norm_position[1])
        angle_bin = round(self.direction_angle / 5) * 5
        area_bin = round(self.area_fraction, 2)
        return f"{self.element_type}|{pos_bin}|{angle_bin}|{area_bin}"


@dataclass
class DetectionResult:
    """Result of watermark detection across a document."""
    watermark_signatures: Set[str] = field(default_factory=set)
    candidates_by_page: Dict[Any, Any] = field(default_factory=dict)
    confidence: float = 0.0
    description: str = ""


# =============================================================================
# WATERMARK DETECTOR BASE CLASS
# =============================================================================

class WatermarkDetector(ABC):
    """Abstract base class for watermark detection strategies."""

    def __init__(self, sample_pages: int = 5, repeat_threshold: float = 0.8):
        self.sample_pages = sample_pages
        self.repeat_threshold = repeat_threshold

    @abstractmethod
    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        pass

    @abstractmethod
    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        pass

    def _get_pages_to_sample(self, doc: fitz.Document) -> List[int]:
        total = len(doc)
        sample_count = min(self.sample_pages, total)
        return list(range(sample_count))

    def _get_hit_threshold(self, sampled_pages: int) -> int:
        return max(1, int(sampled_pages * self.repeat_threshold))


# =============================================================================
# CASE #1: DIAGONAL TEXT WATERMARK DETECTOR
# =============================================================================

class DiagonalTextDetector(WatermarkDetector):
    """Detects diagonal text watermarks in main content streams."""

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_angle: float = 15.0,
                 max_angle: float = 75.0,
                 min_area_fraction: float = 0.02,
                 max_area_fraction: float = 0.5):
        super().__init__(sample_pages, repeat_threshold)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Analyzing pages for diagonal text watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        signature_counts: Counter = Counter()
        candidates_by_page: Dict[int, List[WatermarkCandidate]] = defaultdict(list)
        signature_examples: Dict[str, WatermarkCandidate] = {}

        for page_num in pages_to_sample:
            page = doc.load_page(page_num)
            page_rect = page.rect

            candidates = self._extract_diagonal_text_candidates(page, page_num, page_rect)

            for candidate in candidates:
                candidates_by_page[page_num].append(candidate)
                signature_counts[candidate.signature] += 1

                if candidate.signature not in signature_examples:
                    signature_examples[candidate.signature] = candidate

        watermark_signatures = {
            sig for sig, count in signature_counts.items()
            if count >= hit_threshold
        }

        if status_cb and watermark_signatures:
            for sig in watermark_signatures:
                example = signature_examples.get(sig)
                if example:
                    status_cb(
                        f"Found diagonal watermark: {example.direction_angle:.1f}° angle, "
                        f"{example.area_fraction * 100:.1f}% page area"
                    )

        return DetectionResult(
            watermark_signatures=watermark_signatures,
            candidates_by_page=dict(candidates_by_page),
            confidence=len(watermark_signatures) / max(1, len(signature_counts)) if signature_counts else 0,
            description=f"Found {len(watermark_signatures)} diagonal text watermark pattern(s)"
        )

    def _extract_diagonal_text_candidates(self, page: fitz.Page, page_num: int,
                                          page_rect: fitz.Rect) -> List[WatermarkCandidate]:
        candidates = []

        try:
            text_dict = page.get_text("dict")
        except Exception:
            return candidates

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox = block.get("bbox")
            if not bbox:
                continue

            area_frac = calculate_area_fraction(bbox, page_rect)
            if not (self.min_area_fraction <= area_frac <= self.max_area_fraction):
                continue

            max_angle = 0.0
            font_name = ""
            font_size = 0.0
            color = 0
            text_parts = []

            for line in block.get("lines", []):
                direction = line.get("dir")
                if direction:
                    angle = abs(direction_to_angle(direction))
                    if angle > 90:
                        angle = 180 - angle
                    if angle > max_angle:
                        max_angle = angle

                for span in line.get("spans", []):
                    if not font_name:
                        font_name = span.get("font", "")
                    if font_size == 0:
                        font_size = span.get("size", 0)
                    if color == 0:
                        color = span.get("color", 0)
                    text_parts.append(span.get("text", ""))

            if not (self.min_angle <= max_angle <= self.max_angle):
                continue

            norm_pos = normalize_position(bbox, page_rect)

            candidate = WatermarkCandidate(
                page_num=page_num,
                bbox=bbox,
                element_type="diagonal_text",
                norm_position=norm_pos,
                area_fraction=area_frac,
                text_content=" ".join(text_parts),
                font_name=font_name,
                font_size=font_size,
                direction_angle=max_angle,
                color=color
            )
            candidates.append(candidate)

        return candidates

    def _segment_has_diagonal_text(self, segment: str,
                                   cm_pattern: re.Pattern,
                                   tm_pattern: re.Pattern) -> bool:
        if not RE_DIAGONAL_QUICK.search(segment):
            return False
        
        for pattern in (tm_pattern, cm_pattern):
            for m in pattern.finditer(segment):
                try:
                    a = float(m.group(1))
                    b = float(m.group(2))
                    if is_diagonal_angle(a, b, self.min_angle, self.max_angle):
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    def _remove_diagonal_text_segments_from_stream(self, content: str,
                                                   bt_pattern: re.Pattern,
                                                   cm_pattern: re.Pattern,
                                                   tm_pattern: re.Pattern) -> Tuple[str, int]:
        removed = 0
        out_parts: List[str] = []
        last_end = 0

        for match in bt_pattern.finditer(content):
            start, end = match.span()
            segment = match.group(0)

            if self._segment_has_diagonal_text(segment, cm_pattern, tm_pattern):
                removed += 1
                out_parts.append(content[last_end:start])
                last_end = end

        out_parts.append(content[last_end:])
        return "".join(out_parts), removed

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing diagonal text watermarks...")

        bt_pattern = RE_BT_ET
        cm_pattern = RE_CM_MATRIX
        tm_pattern = RE_TM_MATRIX

        removed_count = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_rect = page.rect

            watermark_regions = self._identify_watermark_regions(
                page, page_num, page_rect, result.watermark_signatures
            )
            if not watermark_regions:
                continue

            xrefs = page.get_contents()
            if not xrefs:
                continue

            if isinstance(xrefs, int):
                xrefs = [xrefs]
            elif isinstance(xrefs, (list, tuple)):
                xrefs = list(xrefs)
            else:
                continue

            page_removed = 0

            for xref in xrefs:
                try:
                    raw_bytes = doc.xref_stream(xref)
                    if raw_bytes is None:
                        continue
                    content = raw_bytes.decode("latin-1", errors="ignore")
                except Exception:
                    continue

                new_content, removed_here = self._remove_diagonal_text_segments_from_stream(
                    content, bt_pattern, cm_pattern, tm_pattern
                )

                if removed_here > 0 and new_content != content:
                    try:
                        doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                        page_removed += removed_here
                    except Exception:
                        pass

            if page_removed > 0:
                removed_count += page_removed

        if status_cb:
            status_cb(f"Removed {removed_count} diagonal text watermark block(s)")

        return removed_count

    def _identify_watermark_regions(self, page: fitz.Page, page_num: int,
                                    page_rect: fitz.Rect,
                                    signatures: Set[str]) -> List[fitz.Rect]:
        regions = []

        try:
            text_dict = page.get_text("dict")
        except Exception:
            return regions

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox = block.get("bbox")
            if not bbox:
                continue

            area_frac = calculate_area_fraction(bbox, page_rect)
            if not (self.min_area_fraction <= area_frac <= self.max_area_fraction):
                continue

            max_angle = 0.0
            for line in block.get("lines", []):
                direction = line.get("dir")
                if direction:
                    angle = abs(direction_to_angle(direction))
                    if angle > 90:
                        angle = 180 - angle
                    if angle > max_angle:
                        max_angle = angle

            if not (self.min_angle <= max_angle <= self.max_angle):
                continue

            norm_pos = normalize_position(bbox, page_rect)
            candidate = WatermarkCandidate(
                page_num=page_num,
                bbox=bbox,
                element_type="diagonal_text",
                norm_position=norm_pos,
                area_fraction=area_frac,
                direction_angle=max_angle
            )

            if candidate.signature in signatures:
                expanded = fitz.Rect(
                    bbox[0] - 10, bbox[1] - 10,
                    bbox[2] + 10, bbox[3] + 10
                )
                regions.append(expanded)

        return regions


# =============================================================================
# CASE #2 & #3: XOBJECT-BASED WATERMARK DETECTOR
# =============================================================================

class XObjectWatermarkDetector(WatermarkDetector):
    """Detects watermarks stored as XObject Forms."""

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_angle: float = 15.0,
                 max_angle: float = 75.0):
        super().__init__(sample_pages, repeat_threshold)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.cm_pattern = RE_CM_MATRIX
        self.tm_pattern = RE_TM_MATRIX

    def _normalize_name(self, name: Any) -> str:
        if isinstance(name, bytes):
            name = name.decode("latin-1", errors="ignore")
        name = str(name)
        return name.lstrip("/")

    def _matrix_is_diagonal(self, a: float, b: float) -> bool:
        return is_diagonal_angle(a, b, self.min_angle, self.max_angle)

    def _stream_has_diagonal(self, content: str) -> bool:
        if not RE_DIAGONAL_QUICK.search(content):
            return False
        
        for pattern in (self.cm_pattern, self.tm_pattern):
            for m in pattern.finditer(content):
                try:
                    a = float(m.group(1))
                    b = float(m.group(2))
                    if self._matrix_is_diagonal(a, b):
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    def _find_diagonal_xobject_invocations(self, doc: fitz.Document, page: fitz.Page) -> Dict[str, int]:
        diagonal_invocations: Dict[str, int] = {}
        
        xrefs = page.get_contents()
        if not xrefs:
            return diagonal_invocations
            
        if isinstance(xrefs, int):
            xrefs = [xrefs]
        
        do_pattern = re.compile(
            r"q\s+[^Q]*?"
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm\s*"
            r"/(\w+)\s+Do"
            r"[^Q]*?Q",
            re.DOTALL
        )
        
        try:
            xobjects = page.get_xobjects()
            name_to_xref = {self._normalize_name(xo[1]): xo[0] for xo in xobjects if len(xo) >= 2}
        except Exception:
            name_to_xref = {}
        
        for xref in xrefs:
            try:
                raw_bytes = doc.xref_stream(xref)
                if raw_bytes is None:
                    continue
                content = raw_bytes.decode("latin-1", errors="ignore")
            except Exception:
                continue
            
            for match in do_pattern.finditer(content):
                try:
                    a = float(match.group(1))
                    b = float(match.group(2))
                    name = match.group(7)
                    
                    if self._matrix_is_diagonal(a, b):
                        xobj_xref = name_to_xref.get(name, 0)
                        if xobj_xref:
                            diagonal_invocations[name] = xobj_xref
                except (ValueError, TypeError, IndexError):
                    continue
        
        return diagonal_invocations

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Analyzing XObject Forms for watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        xref_page_count: Counter = Counter()
        xref_to_names: Dict[int, Set[str]] = defaultdict(set)
        xref_has_internal_diagonal: Dict[int, bool] = {}

        for page_num in pages_to_sample:
            page = doc.load_page(page_num)

            try:
                xobjects = page.get_xobjects()
            except Exception:
                continue

            diagonal_invocations = self._find_diagonal_xobject_invocations(doc, page)
            seen_on_this_page: Set[int] = set()

            for xobj in xobjects:
                if len(xobj) < 2:
                    continue

                xref = xobj[0]
                raw_name = xobj[1]
                name = self._normalize_name(raw_name)

                xref_to_names[xref].add(name)

                if xref not in xref_has_internal_diagonal:
                    is_internal_diag = False
                    try:
                        stream = doc.xref_stream(xref)
                        if stream:
                            content = stream.decode("latin-1", errors="ignore")
                            is_internal_diag = self._stream_has_diagonal(content)
                    except Exception:
                        pass
                    xref_has_internal_diagonal[xref] = is_internal_diag

                is_external_diag = name in diagonal_invocations

                if (xref_has_internal_diagonal.get(xref, False) or is_external_diag):
                    if xref not in seen_on_this_page:
                        xref_page_count[xref] += 1
                        seen_on_this_page.add(xref)

        watermark_xrefs: Set[int] = set()
        for xref, count in xref_page_count.items():
            if count >= hit_threshold:
                watermark_xrefs.add(xref)
                if status_cb:
                    names = ", ".join(sorted(xref_to_names.get(xref, []))) or "<?>"
                    case_type = "internal" if xref_has_internal_diagonal.get(xref, False) else "external"
                    status_cb(
                        f"Found XObject watermark: xref {xref} (names: {names}) "
                        f"on {count} pages ({case_type} rotation)"
                    )

        result = DetectionResult(
            watermark_signatures={str(x) for x in watermark_xrefs},
            confidence=len(watermark_xrefs) / max(1, len(xref_page_count)) if xref_page_count else 0,
            description=f"Found {len(watermark_xrefs)} XObject watermark(s)"
        )

        result.candidates_by_page = {
            "watermark_xrefs": watermark_xrefs,
            "xref_to_names": {xref: list(names) for xref, names in xref_to_names.items()}
        }

        return result

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        candidates = result.candidates_by_page or {}
        watermark_xrefs: Set[int] = set()

        if "watermark_xrefs" in candidates:
            watermark_xrefs = set(candidates["watermark_xrefs"] or [])
        else:
            for sig in result.watermark_signatures:
                try:
                    watermark_xrefs.add(int(sig))
                except Exception:
                    continue

        if not watermark_xrefs:
            return 0

        if status_cb:
            status_cb("Removing XObject watermarks...")

        xref_to_names: Dict[int, List[str]] = candidates.get("xref_to_names", {}) or {}

        removed_streams = 0

        for xref in watermark_xrefs:
            try:
                if doc.xref_stream(xref) is not None:
                    doc.update_stream(xref, b"")
                    removed_streams += 1
            except Exception:
                continue

        do_patterns: List[re.Pattern] = []
        for xref in watermark_xrefs:
            names = xref_to_names.get(xref, [])
            for name in names:
                nm = self._normalize_name(name)
                if not nm:
                    continue
                pat = re.compile(r"/" + re.escape(nm) + r"\s+Do\b")
                do_patterns.append(pat)

        removed_refs = 0

        if do_patterns:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                xrefs = page.get_contents()
                if not xrefs:
                    continue

                if isinstance(xrefs, int):
                    xref_list = [xrefs]
                else:
                    xref_list = list(xrefs)

                for stream_xref in xref_list:
                    try:
                        raw_bytes = doc.xref_stream(stream_xref)
                        if raw_bytes is None:
                            continue
                        content = raw_bytes.decode("latin-1", errors="ignore")
                    except Exception:
                        continue

                    new_content = content
                    modified = False

                    for pattern in do_patterns:
                        new_content, n_subs = pattern.subn("", new_content)
                        if n_subs:
                            removed_refs += n_subs
                            modified = True

                    if modified and new_content != content:
                        try:
                            doc.update_stream(stream_xref, new_content.encode("latin-1", errors="ignore"))
                        except Exception:
                            continue

        total_removed = removed_streams + removed_refs

        if status_cb:
            status_cb(
                f"Removed {removed_streams} XObject stream(s) and "
                f"{removed_refs} Do reference(s)"
            )

        return total_removed


# =============================================================================
# CASE #4: NESTED XOBJECT WATERMARK DETECTOR
# =============================================================================

class NestedXObjectWatermarkDetector(WatermarkDetector):
    """Detects watermarks deeply nested inside XObject chains."""

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_angle: float = 15.0,
                 max_angle: float = 75.0,
                 large_bbox_threshold: float = 10000.0):
        super().__init__(sample_pages, repeat_threshold)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.large_bbox_threshold = large_bbox_threshold
        self.cm_pattern = RE_CM_MATRIX
        self.tm_pattern = RE_TM_MATRIX
        self.do_pattern = RE_DO_CALL
        self.bt_et_pattern = RE_BT_ET

    def _normalize_name(self, name: Any) -> str:
        if isinstance(name, bytes):
            name = name.decode("latin-1", errors="ignore")
        name = str(name)
        return name.lstrip("/")

    def _matrix_is_diagonal(self, a: float, b: float) -> bool:
        return is_diagonal_angle(a, b, self.min_angle, self.max_angle)

    def _has_large_bbox(self, bbox: tuple) -> bool:
        if not bbox or len(bbox) < 4:
            return False
        try:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            return width > self.large_bbox_threshold or height > self.large_bbox_threshold
        except (TypeError, IndexError):
            return False

    def _stream_has_diagonal_text(self, content: str) -> bool:
        if not RE_DIAGONAL_QUICK.search(content):
            return False
        
        for pattern in (self.cm_pattern, self.tm_pattern):
            for m in pattern.finditer(content):
                try:
                    a = float(m.group(1))
                    b = float(m.group(2))
                    if self._matrix_is_diagonal(a, b):
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    def _stream_has_text(self, content: str) -> bool:
        return 'BT' in content and 'ET' in content

    def _scan_all_xrefs_for_watermarks(self, doc: fitz.Document, 
                                        status_cb: Optional[Callable] = None) -> Set[int]:
        watermark_xrefs: Set[int] = set()
        
        try:
            xref_count = doc.xref_length()
        except Exception:
            return watermark_xrefs
        
        diagonal_streams_by_size: Dict[int, List[int]] = defaultdict(list)
        
        for xref in range(1, xref_count):
            try:
                stream = doc.xref_stream(xref)
                if not stream:
                    continue
                
                content = stream.decode("latin-1", errors="ignore")
                stream_len = len(content)
                
                if 'BT' not in content or 'ET' not in content:
                    continue
                
                has_diagonal = self._stream_has_diagonal_text(content)
                if not has_diagonal:
                    continue
                
                size_bucket = (stream_len // 10) * 10
                diagonal_streams_by_size[size_bucket].append(xref)
                
                obj_str = doc.xref_object(xref)
                is_form = '/Subtype /Form' in obj_str or '/Subtype/Form' in obj_str
                
                if is_form:
                    bbox_match = RE_BBOX.search(obj_str)
                    if bbox_match:
                        try:
                            w = abs(float(bbox_match.group(3)) - float(bbox_match.group(1)))
                            h = abs(float(bbox_match.group(4)) - float(bbox_match.group(2)))
                            if w > self.large_bbox_threshold or h > self.large_bbox_threshold:
                                watermark_xrefs.add(xref)
                                if status_cb:
                                    status_cb(f"  Found candidate: xref {xref} (Form with large bbox + diagonal text)")
                                continue
                        except (ValueError, TypeError):
                            pass
                    
                    if stream_len < 2000 and 'Do' not in content:
                        watermark_xrefs.add(xref)
                        if status_cb:
                            status_cb(f"  Found candidate: xref {xref} (small Form with diagonal text)")
                
            except Exception:
                continue
        
        num_pages = len(doc)
        for size_bucket, xref_list in diagonal_streams_by_size.items():
            count = len(xref_list)
            
            is_page_pattern = any(
                abs(count - num_pages * mult) <= num_pages * 0.1
                for mult in (1, 2, 3, 4)
            )
            
            if is_page_pattern and count >= 10:
                watermark_xrefs.update(xref_list)
                if status_cb:
                    status_cb(f"  Found Case #4 pattern: {count} streams of ~{size_bucket} bytes (diagonal text)")
        
        return watermark_xrefs

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning all XObjects for nested watermarks...")

        watermark_xrefs = self._scan_all_xrefs_for_watermarks(doc, status_cb)
        
        if not watermark_xrefs:
            if status_cb:
                status_cb("Pattern detection found nothing, checking PyMuPDF text extraction...")
            
            has_diagonal_text = False
            try:
                page = doc[0]
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        dir_vec = line.get("dir", (1, 0))
                        if len(dir_vec) >= 2:
                            angle = abs(math.degrees(math.atan2(dir_vec[1], dir_vec[0])))
                            if angle > 90:
                                angle = 180 - angle
                            if self.min_angle <= angle <= self.max_angle:
                                has_diagonal_text = True
                                break
                    if has_diagonal_text:
                        break
            except Exception:
                pass
            
            if has_diagonal_text:
                if status_cb:
                    status_cb("PyMuPDF detected diagonal text, performing aggressive scan...")
                watermark_xrefs = self._aggressive_diagonal_scan(doc, status_cb)
        
        if not watermark_xrefs:
            return DetectionResult(description="No nested watermarks found")

        xref_to_names: Dict[int, Set[str]] = defaultdict(set)
        
        for page_num in range(min(5, len(doc))):
            try:
                page = doc.load_page(page_num)
                for xobj in page.get_xobjects():
                    if len(xobj) >= 2 and xobj[0] in watermark_xrefs:
                        xref_to_names[xobj[0]].add(self._normalize_name(xobj[1]))
            except Exception:
                pass
        
        try:
            xref_count = doc.xref_length()
            for xref in range(1, xref_count):
                try:
                    obj_str = doc.xref_object(xref)
                    for wm_xref in watermark_xrefs:
                        pattern = re.compile(r'/(\w+)\s+' + str(wm_xref) + r'\s+0\s+R')
                        for match in pattern.finditer(obj_str):
                            xref_to_names[wm_xref].add(match.group(1))
                except Exception:
                    continue
        except Exception:
            pass

        if status_cb:
            status_cb(f"Found {len(watermark_xrefs)} streams with diagonal text")

        result = DetectionResult(
            watermark_signatures={str(x) for x in watermark_xrefs},
            confidence=1.0 if watermark_xrefs else 0.0,
            description=f"Found {len(watermark_xrefs)} nested XObject watermark(s)"
        )

        result.candidates_by_page = {
            "watermark_xrefs": watermark_xrefs,
            "xref_to_names": {xref: list(names) for xref, names in xref_to_names.items()}
        }

        return result

    def _aggressive_diagonal_scan(self, doc: fitz.Document, 
                                   status_cb: Optional[Callable] = None) -> Set[int]:
        watermark_xrefs: Set[int] = set()
        
        try:
            xref_count = doc.xref_length()
        except Exception:
            return watermark_xrefs
        
        for xref in range(1, xref_count):
            try:
                stream = doc.xref_stream(xref)
                if not stream:
                    continue
                    
                content = stream.decode("latin-1", errors="ignore")
                
                if self._stream_has_diagonal_text(content) and self._stream_has_text(content):
                    watermark_xrefs.add(xref)
                    
            except Exception:
                continue
        
        if status_cb and watermark_xrefs:
            status_cb(f"  Aggressive scan found {len(watermark_xrefs)} diagonal text streams")
        
        return watermark_xrefs

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        candidates = result.candidates_by_page or {}
        watermark_xrefs: Set[int] = set()

        if "watermark_xrefs" in candidates:
            watermark_xrefs = set(candidates["watermark_xrefs"] or [])
        else:
            for sig in result.watermark_signatures:
                try:
                    watermark_xrefs.add(int(sig))
                except Exception:
                    continue

        if not watermark_xrefs:
            return 0

        if status_cb:
            status_cb("Removing nested XObject watermarks (surgical removal)...")

        bt_et_pattern = RE_BT_ET
        removed_count = 0

        for xref in watermark_xrefs:
            try:
                stream = doc.xref_stream(xref)
                if stream is None:
                    continue
                    
                content = stream.decode("latin-1", errors="ignore")
                new_content, blocks_removed = self._remove_diagonal_text_blocks(content, bt_et_pattern)
                
                if blocks_removed > 0 and new_content != content:
                    doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                    removed_count += blocks_removed
                    
            except Exception:
                continue

        if status_cb:
            status_cb("Scanning all streams for remaining diagonal text...")
        
        try:
            xref_count = doc.xref_length()
            for xref in range(1, xref_count):
                if xref in watermark_xrefs:
                    continue
                    
                try:
                    stream = doc.xref_stream(xref)
                    if stream is None:
                        continue
                        
                    content = stream.decode("latin-1", errors="ignore")
                    
                    if 'BT' not in content:
                        continue
                    if not RE_DIAGONAL_QUICK.search(content):
                        continue
                    
                    new_content, blocks_removed = self._remove_diagonal_text_blocks(content, bt_et_pattern)
                    
                    if blocks_removed > 0 and new_content != content:
                        doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                        removed_count += blocks_removed
                        
                except Exception:
                    continue
        except Exception:
            pass

        if status_cb:
            status_cb(f"Surgically removed {removed_count} diagonal text block(s)")

        return removed_count

    def _remove_diagonal_text_blocks(self, content: str, bt_et_pattern: re.Pattern) -> Tuple[str, int]:
        removed = 0
        result_parts = []
        last_end = 0
        
        for match in bt_et_pattern.finditer(content):
            block = match.group(0)
            
            if self._block_has_diagonal_tm(block):
                result_parts.append(content[last_end:match.start()])
                last_end = match.end()
                removed += 1
        
        result_parts.append(content[last_end:])
        return "".join(result_parts), removed

    def _block_has_diagonal_tm(self, block: str) -> bool:
        for match in self.tm_pattern.finditer(block):
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                if self._matrix_is_diagonal(a, b):
                    return True
            except (ValueError, TypeError):
                continue
        
        for match in self.cm_pattern.finditer(block):
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                if self._matrix_is_diagonal(a, b):
                    return True
            except (ValueError, TypeError):
                continue
        
        return False


# =============================================================================
# CASE #5: HORIZONTAL TEXT WATERMARK DETECTOR
# =============================================================================

WATERMARK_PHRASES = {
    'confidential information - do not distribute',
}


class HorizontalTextWatermarkDetector(WatermarkDetector):
    """Detects horizontal text watermarks."""

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 custom_phrases: Optional[Set[str]] = None,
                 margin_threshold: float = 0.1):
        super().__init__(sample_pages, repeat_threshold)
        self.watermark_phrases = WATERMARK_PHRASES.copy()
        if custom_phrases:
            self.watermark_phrases.update(p.lower() for p in custom_phrases)
        self.margin_threshold = margin_threshold
        self.gray_threshold = 40
        self.alpha_threshold = 200

    def _is_gray_color(self, color_int: int) -> bool:
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        return max_diff <= self.gray_threshold

    def _is_in_margin(self, bbox: Tuple[float, float, float, float], 
                      page_height: float) -> bool:
        y0, y1 = bbox[1], bbox[3]
        top_margin = page_height * self.margin_threshold
        bottom_margin = page_height * (1 - self.margin_threshold)
        return y1 <= top_margin or y0 >= bottom_margin

    def _text_contains_watermark_phrase(self, text: str) -> bool:
        text_lower = text.lower()
        for phrase in self.watermark_phrases:
            if phrase in text_lower:
                return True
        return False

    def _is_horizontal(self, direction: Tuple[float, float]) -> bool:
        if not direction or len(direction) < 2:
            return False
        dx, dy = abs(direction[0]), abs(direction[1])
        return dx > 0.98 and dy < 0.2

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning for horizontal text watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        phrase_signatures: Counter = Counter()
        visual_signatures: Counter = Counter()
        signature_examples: Dict[str, dict] = {}

        for page_num in pages_to_sample:
            page = doc.load_page(page_num)
            page_rect = page.rect
            page_height = page_rect.height

            try:
                text_dict = page.get_text("dict")
            except Exception:
                continue

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                block_text_parts = []
                is_horizontal = True
                is_gray = False
                is_transparent = False
                
                for line in block.get("lines", []):
                    direction = line.get("dir", (1, 0))
                    if not self._is_horizontal(direction):
                        is_horizontal = False
                        break
                    
                    for span in line.get("spans", []):
                        block_text_parts.append(span.get("text", ""))
                        
                        color = span.get("color", 0)
                        if self._is_gray_color(color):
                            is_gray = True
                        
                        alpha = span.get("alpha", 255)
                        if alpha is not None and alpha < self.alpha_threshold:
                            is_transparent = True

                if not is_horizontal:
                    continue

                block_text = " ".join(block_text_parts)
                norm_pos = normalize_position(bbox, page_rect)
                pos_bin = position_bin(norm_pos[0], norm_pos[1], granularity=5)
                
                if self._text_contains_watermark_phrase(block_text):
                    sig = f"phrase_{pos_bin[0]}_{pos_bin[1]}"
                    phrase_signatures[sig] += 1
                    if sig not in signature_examples:
                        signature_examples[sig] = {
                            "text": block_text[:50],
                            "bbox": bbox,
                            "method": "phrase",
                            "page": page_num
                        }
                
                elif is_gray and is_transparent and self._is_in_margin(bbox, page_height):
                    width_bin = int(bbox[2] - bbox[0]) // 10
                    height_bin = int(bbox[3] - bbox[1]) // 5
                    sig = f"visual_{pos_bin[0]}_{pos_bin[1]}_{width_bin}_{height_bin}"
                    visual_signatures[sig] += 1
                    if sig not in signature_examples:
                        signature_examples[sig] = {
                            "text": block_text[:50],
                            "bbox": bbox,
                            "method": "visual",
                            "page": page_num
                        }

        watermark_signatures = set()
        
        for sig, count in phrase_signatures.items():
            if count >= hit_threshold:
                watermark_signatures.add(sig)
                if status_cb:
                    example = signature_examples.get(sig, {})
                    status_cb(f"Found phrase watermark: '{example.get('text', '')[:30]}...'")
        
        for sig, count in visual_signatures.items():
            if count >= hit_threshold:
                watermark_signatures.add(sig)
                if status_cb:
                    example = signature_examples.get(sig, {})
                    status_cb(f"Found visual watermark (gray/transparent): '{example.get('text', '')[:30]}...'")

        result = DetectionResult(
            watermark_signatures=watermark_signatures,
            confidence=len(watermark_signatures) / max(1, len(phrase_signatures) + len(visual_signatures)) if (phrase_signatures or visual_signatures) else 0,
            description=f"Found {len(watermark_signatures)} horizontal text watermark pattern(s)"
        )
        
        result.candidates_by_page = {
            "signatures": watermark_signatures,
            "examples": signature_examples
        }

        return result

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing horizontal text watermarks...")

        candidates = result.candidates_by_page or {}
        examples = candidates.get("examples", {})
        
        phrase_bboxes = []
        visual_bboxes = []
        
        for sig in result.watermark_signatures:
            example = examples.get(sig, {})
            bbox = example.get("bbox")
            method = example.get("method")
            if bbox:
                if method == "phrase":
                    phrase_bboxes.append(bbox)
                else:
                    visual_bboxes.append(bbox)

        removed_count = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_rect = page.rect
            page_height = page_rect.height

            try:
                text_dict = page.get_text("dict")
            except Exception:
                continue

            watermark_rects = []

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                block_text_parts = []
                is_gray = False
                is_transparent = False
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text_parts.append(span.get("text", ""))
                        
                        color = span.get("color", 0)
                        if self._is_gray_color(color):
                            is_gray = True
                        
                        alpha = span.get("alpha", 255)
                        if alpha is not None and alpha < self.alpha_threshold:
                            is_transparent = True

                block_text = " ".join(block_text_parts)
                should_remove = False
                
                if self._text_contains_watermark_phrase(block_text):
                    for ref_bbox in phrase_bboxes:
                        if (abs(bbox[0] - ref_bbox[0]) < 30 and 
                            abs(bbox[1] - ref_bbox[1]) < 30):
                            should_remove = True
                            break
                
                if not should_remove and is_gray and is_transparent and self._is_in_margin(bbox, page_height):
                    for ref_bbox in visual_bboxes:
                        if (abs(bbox[0] - ref_bbox[0]) < 20 and 
                            abs(bbox[1] - ref_bbox[1]) < 20 and
                            abs(bbox[2] - ref_bbox[2]) < 20 and
                            abs(bbox[3] - ref_bbox[3]) < 20):
                            should_remove = True
                            break

                if should_remove:
                    watermark_rects.append(fitz.Rect(bbox))

            if watermark_rects:
                for rect in watermark_rects:
                    expanded = fitz.Rect(
                        rect.x0 - 2, rect.y0 - 2,
                        rect.x1 + 2, rect.y1 + 2
                    )
                    page.add_redact_annot(expanded, fill=False)
                    removed_count += 1

                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        if status_cb:
            status_cb(f"Removed {removed_count} horizontal text watermark(s)")

        return removed_count


# =============================================================================
# CASE #6: VECTOR/PATH-BASED WATERMARK DETECTOR
# =============================================================================

class VectorWatermarkDetector(WatermarkDetector):
    """Detects watermarks rendered as vector paths instead of text."""

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_opacity: float = 0.05,
                 max_opacity: float = 0.35):
        super().__init__(sample_pages, repeat_threshold)
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity

    def _is_watermark_drawing(self, drawing: dict) -> Tuple[bool, float]:
        fill_opacity = drawing.get("fill_opacity")
        if fill_opacity is None:
            return False, 0.0
        
        if not (self.min_opacity <= fill_opacity <= self.max_opacity):
            return False, 0.0
        
        fill = drawing.get("fill")
        if fill is None:
            return False, 0.0
        
        return True, fill_opacity

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning for vector/path-based watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        opacity_counts_per_page: Dict[int, Counter] = {}
        all_watermark_drawings: Dict[int, List[dict]] = {}
        
        for page_num in pages_to_sample:
            page = doc.load_page(page_num)

            try:
                drawings = page.get_drawings()
            except Exception:
                continue

            page_drawings = []
            opacity_counter: Counter = Counter()

            for drawing in drawings:
                is_wm, fill_opacity = self._is_watermark_drawing(drawing)
                if not is_wm:
                    continue

                rect = drawing.get("rect")
                if not rect:
                    continue

                opacity_bin = int(fill_opacity * 100)
                opacity_counter[opacity_bin] += 1
                
                page_drawings.append({
                    "rect": rect,
                    "fill_opacity": fill_opacity,
                    "opacity_bin": opacity_bin,
                    "fill": drawing.get("fill"),
                })

            opacity_counts_per_page[page_num] = opacity_counter
            all_watermark_drawings[page_num] = page_drawings

        watermark_opacity_bins: Set[int] = set()
        
        if len(opacity_counts_per_page) >= hit_threshold:
            all_bins: Set[int] = set()
            for counter in opacity_counts_per_page.values():
                all_bins.update(counter.keys())
            
            for opacity_bin in all_bins:
                pages_with_bin = sum(
                    1 for counter in opacity_counts_per_page.values()
                    if counter[opacity_bin] > 0
                )
                
                if pages_with_bin >= hit_threshold:
                    counts = [
                        counter[opacity_bin] 
                        for counter in opacity_counts_per_page.values()
                        if counter[opacity_bin] > 0
                    ]
                    if counts:
                        avg_count = sum(counts) / len(counts)
                        if avg_count >= 3:
                            watermark_opacity_bins.add(opacity_bin)
                            if status_cb:
                                status_cb(f"Found vector watermark: ~{opacity_bin}% opacity, ~{avg_count:.0f} paths/page")

        if not watermark_opacity_bins:
            return DetectionResult(description="No vector watermarks found")

        result = DetectionResult(
            watermark_signatures={str(b) for b in watermark_opacity_bins},
            confidence=1.0 if watermark_opacity_bins else 0.0,
            description=f"Found {len(watermark_opacity_bins)} vector watermark pattern(s)"
        )

        result.candidates_by_page = {
            "opacity_bins": watermark_opacity_bins,
            "drawings_by_page": all_watermark_drawings
        }

        return result

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing vector/path-based watermarks (surgical removal)...")

        candidates = result.candidates_by_page or {}
        watermark_opacity_bins: Set[int] = set()
        
        for sig in result.watermark_signatures:
            try:
                watermark_opacity_bins.add(int(sig))
            except ValueError:
                continue
        
        if "opacity_bins" in candidates:
            watermark_opacity_bins.update(candidates["opacity_bins"])

        if not watermark_opacity_bins:
            return 0

        watermark_opacities = {b / 100.0 for b in watermark_opacity_bins}
        watermark_gs_names = self._find_watermark_graphics_states(doc, watermark_opacities)
        
        if status_cb:
            status_cb(f"Found {len(watermark_gs_names)} graphics states with watermark opacity")

        if not watermark_gs_names:
            return 0

        removed_count = 0

        try:
            xref_count = doc.xref_length()
        except Exception:
            return 0

        for xref in range(1, xref_count):
            try:
                stream = doc.xref_stream(xref)
                if stream is None:
                    continue
                    
                content = stream.decode("latin-1", errors="ignore")
                
                has_gs = any(f'/{gs} gs' in content or f'/{gs}\ngs' in content or f'/{gs} \ngs' in content 
                            for gs in watermark_gs_names)
                has_fill = 'f*' in content or (' f ' in content) or ('\nf\n' in content) or content.endswith(' f') or content.endswith('\nf')
                
                if not (has_gs and has_fill):
                    continue
                
                new_content, blocks_removed = self._remove_watermark_path_blocks(
                    content, watermark_gs_names
                )
                
                if blocks_removed > 0 and new_content != content:
                    doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                    removed_count += blocks_removed
                    
            except Exception:
                continue

        if status_cb:
            status_cb(f"Surgically removed {removed_count} vector watermark block(s)")

        return removed_count

    def _find_watermark_graphics_states(self, doc: fitz.Document, 
                                         target_opacities: Set[float]) -> Set[str]:
        gs_names: Set[str] = set()
        
        try:
            xref_count = doc.xref_length()
        except Exception:
            return gs_names
        
        ca_pattern = re.compile(r'/ca\s+([\d.]+)')
        gs_xref_to_opacity: Dict[int, float] = {}
        
        for xref in range(1, xref_count):
            try:
                obj_str = doc.xref_object(xref)
                if '/Type /ExtGState' not in obj_str and '/Type/ExtGState' not in obj_str:
                    if '/ca ' not in obj_str and '/ca\n' not in obj_str:
                        continue
                
                match = ca_pattern.search(obj_str)
                if match:
                    opacity = float(match.group(1))
                    gs_xref_to_opacity[xref] = opacity
                    
            except Exception:
                continue
        
        matching_xrefs: Set[int] = set()
        for xref, opacity in gs_xref_to_opacity.items():
            for target in target_opacities:
                if abs(opacity - target) < 0.02:
                    matching_xrefs.add(xref)
                    break
        
        if not matching_xrefs:
            return gs_names
        
        for xref in range(1, xref_count):
            try:
                obj_str = doc.xref_object(xref)
                if '/ExtGState' not in obj_str:
                    continue
                
                for gs_xref in matching_xrefs:
                    pattern = re.compile(r'/(\w+)\s+' + str(gs_xref) + r'\s+0\s+R')
                    for match in pattern.finditer(obj_str):
                        gs_names.add(match.group(1))
                        
            except Exception:
                continue
        
        return gs_names

    def _remove_watermark_path_blocks(self, content: str, 
                                       gs_names: Set[str]) -> Tuple[str, int]:
        if not gs_names:
            return content, 0
        
        gs_pattern = '|'.join(re.escape(name) for name in gs_names)
        
        block_pattern = re.compile(
            r'q\s+'
            r'(?:[^Q]*?)'
            r'/(' + gs_pattern + r')\s+gs'
            r'(?:[^Q]*?)'
            r'(?:f\*|(?<![a-zA-Z])f(?![a-zA-Z]))'
            r'(?:[^Q]*?)'
            r'Q',
            re.DOTALL
        )
        
        removed = 0
        result_parts = []
        last_end = 0
        
        for match in block_pattern.finditer(content):
            block = match.group(0)
            has_path_ops = bool(re.search(r'(?<![a-zA-Z])[mlcvyh](?![a-zA-Z])', block))
            
            if has_path_ops:
                result_parts.append(content[last_end:match.start()])
                last_end = match.end()
                removed += 1
        
        result_parts.append(content[last_end:])
        return "".join(result_parts), removed


# =============================================================================
# WATERMARK REMOVER - MAIN CLASS
# =============================================================================

class WatermarkRemover:
    """Main class that orchestrates watermark detection and removal."""

    def __init__(self):
        self.detectors: List[WatermarkDetector] = []
        self._setup_default_detectors()

    def _setup_default_detectors(self):
        self.detectors.append(DiagonalTextDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0,
            min_area_fraction=0.02,
            max_area_fraction=0.5
        ))
        
        self.detectors.append(XObjectWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0
        ))
        
        self.detectors.append(NestedXObjectWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0,
            large_bbox_threshold=10000.0
        ))
        
        self.detectors.append(HorizontalTextWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7
        ))
        
        self.detectors.append(VectorWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_opacity=0.05,
            max_opacity=0.35
        ))

    def remove_watermarks(self, input_path: str, output_path: str,
                          status_cb: Optional[Callable] = None) -> int:
        try:
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            for detector in self.detectors:
                detector_name = detector.__class__.__name__
                if status_cb:
                    status_cb(f"Running {detector_name}...")

                result = detector.detect(doc, status_cb)

                if result.watermark_signatures:
                    if status_cb:
                        status_cb(result.description)

                    removed = detector.remove(doc, result, status_cb)
                    total_removed += removed

            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()

            if status_cb:
                status_cb(f"Complete: removed {total_removed} watermark elements")

            return total_removed

        except Exception as e:
            raise RuntimeError(f"Watermark removal failed: {str(e)}")


# =============================================================================
# PyQt6 GUI IMPLEMENTATION
# =============================================================================

if USING_PYQT6:
    
    class ProcessorThread(QThread):
        """Thread for processing PDFs."""
        progress_updated = pyqtSignal(int)
        file_processed = pyqtSignal(str, str)
        error_occurred = pyqtSignal(str, str)
        status_updated = pyqtSignal(str)
        processing_complete = pyqtSignal(list)
        preview_ready = pyqtSignal(str, str)

        def __init__(self, pdf_files: List[str]):
            super().__init__()
            self.pdf_files = pdf_files
            self.cancel_requested = False
            self.processed_files = []
            self.remover = WatermarkRemover()

        def run(self):
            total = len(self.pdf_files)

            for i, pdf_file in enumerate(self.pdf_files):
                if self.cancel_requested:
                    break

                try:
                    self.status_updated.emit(f"Processing {os.path.basename(pdf_file)}...")

                    input_path = Path(pdf_file)
                    output_path = input_path.parent / f"(No Watermarks) {input_path.name}"

                    self.remover.remove_watermarks(
                        str(input_path),
                        str(output_path),
                        status_cb=lambda msg: self.status_updated.emit(msg)
                    )

                    self.file_processed.emit(pdf_file, str(output_path))
                    self.processed_files.append((pdf_file, str(output_path)))

                    if i == 0:
                        self.preview_ready.emit(pdf_file, str(output_path))

                    progress = int(((i + 1) / total) * 100)
                    self.progress_updated.emit(progress)

                except Exception as e:
                    self.error_occurred.emit(pdf_file, str(e))

            self.processing_complete.emit(self.processed_files)

        def cancel(self):
            self.cancel_requested = True


    class PreviewThread(QThread):
        """Thread for generating preview."""
        preview_ready = pyqtSignal(str, str)
        error_occurred = pyqtSignal(str, str)
        status_updated = pyqtSignal(str)

        def __init__(self, pdf_file: str):
            super().__init__()
            self.pdf_file = pdf_file
            self.cancel_requested = False
            self.remover = WatermarkRemover()

        def run(self):
            try:
                base = os.path.basename(self.pdf_file)
                output_path = os.path.join(tempfile.gettempdir(), f"(Preview) {base}")

                self.remover.remove_watermarks(
                    self.pdf_file,
                    output_path,
                    status_cb=lambda msg: self.status_updated.emit(f"[Preview] {msg}")
                )

                if not self.cancel_requested:
                    self.preview_ready.emit(self.pdf_file, output_path)

            except Exception as e:
                if not self.cancel_requested:
                    self.error_occurred.emit(self.pdf_file, str(e))

        def cancel(self):
            self.cancel_requested = True


    class ClickableLabel(QLabel):
        """Label that emits clicked signal."""
        clicked = pyqtSignal()

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit()
            super().mousePressEvent(event)


    class PDFPreviewWidget(QScrollArea):
        """Widget for showing original vs processed PDF preview."""

        def __init__(self):
            super().__init__()
            self.setWidgetResizable(True)
            self.setMinimumHeight(400)

            container = QWidget()
            self.setWidget(container)
            self.layout = QVBoxLayout(container)

            self.original_label = ClickableLabel("Original PDF will appear here")
            self.processed_label = ClickableLabel("Processed PDF will appear here")

            for label in [self.original_label, self.processed_label]:
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumHeight(300)
                label.setStyleSheet("""
                    QLabel {
                        border: 2px dashed #aaa;
                        border-radius: 10px;
                        background-color: #f9f9f9;
                        color: #666;
                        font-size: 14px;
                    }
                """)

            self.layout.addWidget(QLabel("Original PDF:"))
            self.layout.addWidget(self.original_label)
            self.layout.addWidget(QLabel("Processed (Preview):"))
            self.layout.addWidget(self.processed_label)

            self.current_pdfs = (None, None)
            self.original_label.clicked.connect(
                lambda: open_file_cross_platform(self.current_pdfs[0])
            )
            self.processed_label.clicked.connect(
                lambda: open_file_cross_platform(self.current_pdfs[1])
            )

        def clear_preview(self, text="Refreshing preview..."):
            self.current_pdfs = (None, None)
            self.original_label.setPixmap(QPixmap())
            self.original_label.setText(text)
            self.processed_label.setPixmap(QPixmap())
            self.processed_label.setText(text)

        def update_preview(self, original_path: str, processed_path: str):
            self.current_pdfs = (original_path, processed_path)

            try:
                orig_pixmap = self._render_page(original_path)
                if orig_pixmap:
                    self.original_label.setPixmap(orig_pixmap.scaled(
                        400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))
                else:
                    self.original_label.setText("Failed to render original")

                if processed_path:
                    proc_pixmap = self._render_page(processed_path)
                    if proc_pixmap:
                        self.processed_label.setPixmap(proc_pixmap.scaled(
                            400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        ))
                    else:
                        self.processed_label.setText("Failed to render processed")
                else:
                    self.processed_label.setText("Processing...")

            except Exception as e:
                print(f"Preview error: {e}")

        def _render_page(self, pdf_path: str, page_num: int = 0) -> Optional[QPixmap]:
            try:
                if not pdf_path or not os.path.exists(pdf_path):
                    return None

                doc = fitz.open(pdf_path)
                if page_num >= len(doc):
                    page_num = 0

                page = doc.load_page(page_num)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                data = pix.tobytes("png")

                pixmap = QPixmap()
                pixmap.loadFromData(data)
                doc.close()

                return pixmap
            except Exception:
                return None


    class DropZoneWidget(QLabel):
        """Drag-and-drop zone for PDF files."""
        files_dropped = pyqtSignal(list)

        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumHeight(150)
            self._set_default_style()
            self.setText("📁 Drag and drop PDF files here\n\nOr click to browse...")
            self.mousePressEvent = self.open_file_dialog

        def _set_default_style(self):
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #007ACC;
                    border-radius: 15px;
                    background-color: #f0f8ff;
                    color: #007ACC;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

        def _set_hover_style(self):
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #28a745;
                    border-radius: 15px;
                    background-color: #e8f5e8;
                    color: #28a745;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                pdf_files = [
                    url.toLocalFile() for url in event.mimeData().urls()
                    if url.toLocalFile().lower().endswith('.pdf')
                ]
                if pdf_files:
                    event.acceptProposedAction()
                    self._set_hover_style()
            else:
                event.ignore()

        def dragLeaveEvent(self, event):
            self._set_default_style()

        def dropEvent(self, event: QDropEvent):
            if event.mimeData().hasUrls():
                pdf_files = [
                    url.toLocalFile() for url in event.mimeData().urls()
                    if url.toLocalFile().lower().endswith('.pdf')
                ]
                if pdf_files:
                    self.files_dropped.emit(pdf_files)
                    event.acceptProposedAction()
            self._set_default_style()

        def open_file_dialog(self, event):
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select PDF Files", "", "PDF Files (*.pdf)"
            )
            if files:
                self.files_dropped.emit(files)


    class WatermarkRemovalApp(QMainWindow):
        """Main application window (PyQt6)."""

        def __init__(self):
            super().__init__()
            self.processor_thread = None
            self.preview_thread = None
            self.current_files = []
            self.processed_files = []

            self.init_ui()
            self.setWindowTitle("PDF Watermark Remover")
            self.setGeometry(100, 100, 1200, 760)

        def init_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setSpacing(15)
            layout.setContentsMargins(20, 20, 20, 20)

            title = QLabel("PDF Watermark Removal Tool")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_font = QFont()
            title_font.setPointSize(20)
            title_font.setBold(True)
            title.setFont(title_font)
            title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
            layout.addWidget(title)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(splitter)

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)

            self.drop_zone = DropZoneWidget()
            self.drop_zone.files_dropped.connect(self.handle_files_dropped)
            left_layout.addWidget(self.drop_zone)

            progress_group = QGroupBox("Processing Progress")
            progress_layout = QVBoxLayout(progress_group)

            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            progress_layout.addWidget(self.progress_bar)

            button_layout = QHBoxLayout()

            self.start_button = QPushButton("Start Processing")
            self.start_button.setEnabled(False)
            self.start_button.clicked.connect(self.start_processing)
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #007ACC;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #005fa3; }
                QPushButton:disabled { background-color: #cccccc; }
            """)

            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.setEnabled(False)
            self.cancel_button.clicked.connect(self.cancel_processing)
            self.cancel_button.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #c82333; }
                QPushButton:disabled { background-color: #cccccc; }
            """)

            button_layout.addWidget(self.start_button)
            button_layout.addWidget(self.cancel_button)
            button_layout.addStretch()
            progress_layout.addLayout(button_layout)
            left_layout.addWidget(progress_group)

            self.status_label = QLabel("Ready to process PDF files")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            left_layout.addWidget(self.status_label)

            self.log_text = QTextEdit()
            self.log_text.setMaximumHeight(240)
            self.log_text.setPlaceholderText("Processing log will appear here...")
            left_layout.addWidget(self.log_text)

            splitter.addWidget(left_panel)

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)

            preview_label = QLabel("PDF Preview")
            preview_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            right_layout.addWidget(preview_label)

            self.preview_widget = PDFPreviewWidget()
            right_layout.addWidget(self.preview_widget)

            splitter.addWidget(right_panel)
            splitter.setSizes([560, 640])

        def handle_files_dropped(self, files: List[str]):
            if self.preview_thread and self.preview_thread.isRunning():
                self.preview_thread.cancel()

            self.current_files = files
            self.processed_files = []
            self.log_text.clear()

            count = len(files)
            self.drop_zone.setText(f"📁 {count} PDF file{'s' if count != 1 else ''} selected\n\nReady to process!")
            self.start_button.setEnabled(True)

            self.log_text.append(f"Selected {count} PDF file(s):")
            for f in files[:10]:
                self.log_text.append(f"  • {os.path.basename(f)}")
            if count > 10:
                self.log_text.append(f"  ... and {count - 10} more files")

            self.status_label.setText(f"Ready to process {count} PDF file(s)")

            if count == 1:
                self.preview_widget.update_preview(files[0], None)
                self._start_preview(files[0])
            else:
                self.preview_widget.clear_preview("Preview disabled for multiple files")

        def _start_preview(self, pdf_file: str):
            self.preview_thread = PreviewThread(pdf_file)
            self.preview_thread.status_updated.connect(self.status_label.setText)
            self.preview_thread.error_occurred.connect(self._on_preview_error)
            self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
            self.preview_thread.start()
            self.log_text.append("🔍 Generating preview...")

        def _on_preview_error(self, file_path: str, error: str):
            self.log_text.append(f"❌ Preview error: {error}")

        def start_processing(self):
            if not self.current_files:
                return

            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            self.processor_thread = ProcessorThread(self.current_files)
            self.processor_thread.progress_updated.connect(self.progress_bar.setValue)
            self.processor_thread.status_updated.connect(self.status_label.setText)
            self.processor_thread.file_processed.connect(self._on_file_processed)
            self.processor_thread.error_occurred.connect(self._on_error)
            self.processor_thread.processing_complete.connect(self._on_complete)
            self.processor_thread.preview_ready.connect(self.preview_widget.update_preview)

            self.processor_thread.start()
            self.log_text.append("\n--- Processing Started ---")

        def cancel_processing(self):
            if self.processor_thread:
                self.processor_thread.cancel()
                self.cancel_button.setText("Cancelling...")
                self.cancel_button.setEnabled(False)

        def _on_file_processed(self, original: str, output: str):
            self.log_text.append(f"✅ Processed: {os.path.basename(original)}")
            self.log_text.append(f"   Output: {os.path.basename(output)}")
            self.processed_files.append((original, output))

        def _on_error(self, file_path: str, error: str):
            self.log_text.append(f"❌ Error: {os.path.basename(file_path)}: {error}")

        def _on_complete(self, processed: List[Tuple[str, str]]):
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancel")
            self.progress_bar.setValue(100)

            success = len(processed)
            total = len(self.current_files)

            self.log_text.append("\n--- Processing Complete ---")
            self.log_text.append(f"Successfully processed: {success}/{total} files")
            self.status_label.setText(f"Complete! Processed {success}/{total} files")

            if success > 0:
                QMessageBox.information(
                    self, "Processing Complete",
                    f"Successfully processed {success} of {total} PDF files."
                )

                if success > 1:
                    resp = QMessageBox.question(
                        self, "Open Files",
                        "Open all processed PDFs?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if resp == QMessageBox.StandardButton.Yes:
                        for _, out in processed:
                            open_file_cross_platform(out)


# =============================================================================
# TKINTER GUI IMPLEMENTATION (FALLBACK)
# =============================================================================

if USING_TKINTER:
    
    class TkinterWatermarkApp:
        """Main application window (tkinter fallback)."""
        
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("PDF Watermark Remover")
            self.root.geometry("1000x700")
            self.root.minsize(800, 600)
            
            self.current_files: List[str] = []
            self.processed_files: List[Tuple[str, str]] = []
            self.processing = False
            self.cancel_requested = False
            self.remover = WatermarkRemover()
            
            self._setup_styles()
            self._create_widgets()
        
        def _setup_styles(self):
            """Configure ttk styles."""
            style = ttk.Style()
            style.theme_use('clam')
            
            # Custom button styles
            style.configure('Start.TButton', 
                          font=('Arial', 10, 'bold'),
                          padding=10)
            style.configure('Cancel.TButton',
                          font=('Arial', 10, 'bold'),
                          padding=10)
        
        def _create_widgets(self):
            """Create all GUI widgets."""
            # Main container with padding
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = ttk.Label(
                main_frame, 
                text="PDF Watermark Removal Tool",
                font=('Arial', 18, 'bold')
            )
            title_label.pack(pady=(0, 15))
            
            # Create paned window for left/right split
            paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True)
            
            # Left panel
            left_frame = ttk.Frame(paned, padding="5")
            paned.add(left_frame, weight=1)
            
            # Drop zone / file selection
            drop_frame = ttk.LabelFrame(left_frame, text="File Selection", padding="10")
            drop_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.drop_label = ttk.Label(
                drop_frame,
                text="Click to select PDF files\n\n(Drag & drop not available in tkinter)",
                font=('Arial', 12),
                anchor='center',
                justify='center'
            )
            self.drop_label.pack(fill=tk.X, pady=20)
            self.drop_label.bind('<Button-1>', self._open_file_dialog)
            
            # Make label look clickable
            self.drop_label.configure(cursor='hand2')
            
            # Progress section
            progress_frame = ttk.LabelFrame(left_frame, text="Processing Progress", padding="10")
            progress_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                progress_frame, 
                variable=self.progress_var,
                maximum=100
            )
            self.progress_bar.pack(fill=tk.X, pady=(0, 10))
            
            # Buttons
            button_frame = ttk.Frame(progress_frame)
            button_frame.pack(fill=tk.X)
            
            self.start_button = ttk.Button(
                button_frame,
                text="Start Processing",
                command=self._start_processing,
                style='Start.TButton',
                state='disabled'
            )
            self.start_button.pack(side=tk.LEFT, padx=(0, 10))
            
            self.cancel_button = ttk.Button(
                button_frame,
                text="Cancel",
                command=self._cancel_processing,
                style='Cancel.TButton',
                state='disabled'
            )
            self.cancel_button.pack(side=tk.LEFT)
            
            # Status label
            self.status_var = tk.StringVar(value="Ready to process PDF files")
            status_label = ttk.Label(
                left_frame,
                textvariable=self.status_var,
                font=('Arial', 10, 'bold'),
                foreground='#28a745'
            )
            status_label.pack(fill=tk.X, pady=(0, 10))
            
            # Log text
            log_frame = ttk.LabelFrame(left_frame, text="Processing Log", padding="5")
            log_frame.pack(fill=tk.BOTH, expand=True)
            
            self.log_text = scrolledtext.ScrolledText(
                log_frame,
                height=10,
                font=('Courier', 9)
            )
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # Right panel - Preview
            right_frame = ttk.Frame(paned, padding="5")
            paned.add(right_frame, weight=1)
            
            preview_label = ttk.Label(
                right_frame,
                text="PDF Preview",
                font=('Arial', 14, 'bold')
            )
            preview_label.pack(pady=(0, 10))
            
            # Preview area
            preview_container = ttk.Frame(right_frame)
            preview_container.pack(fill=tk.BOTH, expand=True)
            
            # Original preview
            orig_frame = ttk.LabelFrame(preview_container, text="Original PDF", padding="5")
            orig_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            self.original_preview = ttk.Label(
                orig_frame,
                text="Original PDF will appear here\n(Click to open)",
                anchor='center',
                cursor='hand2'
            )
            self.original_preview.pack(fill=tk.BOTH, expand=True)
            self.original_preview.bind('<Button-1>', lambda e: self._open_original())
            
            # Processed preview
            proc_frame = ttk.LabelFrame(preview_container, text="Processed (Preview)", padding="5")
            proc_frame.pack(fill=tk.BOTH, expand=True)
            
            self.processed_preview = ttk.Label(
                proc_frame,
                text="Processed PDF will appear here\n(Click to open)",
                anchor='center',
                cursor='hand2'
            )
            self.processed_preview.pack(fill=tk.BOTH, expand=True)
            self.processed_preview.bind('<Button-1>', lambda e: self._open_processed())
            
            # Store preview paths
            self.preview_original_path = None
            self.preview_processed_path = None
            
            # Store PhotoImage references to prevent garbage collection
            self._preview_images = []
        
        def _open_file_dialog(self, event=None):
            """Open file dialog to select PDFs."""
            files = filedialog.askopenfilenames(
                title="Select PDF Files",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
            )
            if files:
                self._handle_files(list(files))
        
        def _handle_files(self, files: List[str]):
            """Handle selected files."""
            self.current_files = files
            self.processed_files = []
            self.log_text.delete(1.0, tk.END)
            
            count = len(files)
            self.drop_label.configure(
                text=f"{count} PDF file{'s' if count != 1 else ''} selected\n\nClick to change selection"
            )
            self.start_button.configure(state='normal')
            
            self._log(f"Selected {count} PDF file(s):")
            for f in files[:10]:
                self._log(f"  • {os.path.basename(f)}")
            if count > 10:
                self._log(f"  ... and {count - 10} more files")
            
            self.status_var.set(f"Ready to process {count} PDF file(s)")
            
            # Generate preview for single file
            if count == 1:
                self._update_preview(files[0], None)
                self._generate_preview_async(files[0])
            else:
                self._clear_preview("Preview disabled for multiple files")
        
        def _log(self, message: str):
            """Add message to log."""
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        
        def _start_processing(self):
            """Start processing files in a background thread."""
            if not self.current_files or self.processing:
                return
            
            self.processing = True
            self.cancel_requested = False
            self.start_button.configure(state='disabled')
            self.cancel_button.configure(state='normal')
            self.progress_var.set(0)
            
            self._log("\n--- Processing Started ---")
            
            # Run in background thread
            thread = threading.Thread(target=self._process_files, daemon=True)
            thread.start()
        
        def _process_files(self):
            """Process files (runs in background thread)."""
            total = len(self.current_files)
            
            for i, pdf_file in enumerate(self.current_files):
                if self.cancel_requested:
                    break
                
                try:
                    self._update_status(f"Processing {os.path.basename(pdf_file)}...")
                    
                    input_path = Path(pdf_file)
                    output_path = input_path.parent / f"(No Watermarks) {input_path.name}"
                    
                    self.remover.remove_watermarks(
                        str(input_path),
                        str(output_path),
                        status_cb=lambda msg: self._update_status(msg)
                    )
                    
                    self.processed_files.append((pdf_file, str(output_path)))
                    self._log_threadsafe(f"✓ Processed: {os.path.basename(pdf_file)}")
                    self._log_threadsafe(f"   Output: {os.path.basename(str(output_path))}")
                    
                    if i == 0:
                        self.root.after(0, lambda p=pdf_file, o=str(output_path): self._update_preview(p, o))
                    
                except Exception as e:
                    self._log_threadsafe(f"✗ Error: {os.path.basename(pdf_file)}: {str(e)}")
                
                progress = ((i + 1) / total) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            # Complete
            self.root.after(0, self._on_complete)
        
        def _update_status(self, message: str):
            """Update status label (thread-safe)."""
            self.root.after(0, lambda: self.status_var.set(message))
        
        def _log_threadsafe(self, message: str):
            """Log message (thread-safe)."""
            self.root.after(0, lambda: self._log(message))
        
        def _cancel_processing(self):
            """Request cancellation."""
            self.cancel_requested = True
            self.cancel_button.configure(state='disabled', text="Cancelling...")
        
        def _on_complete(self):
            """Called when processing is complete."""
            self.processing = False
            self.start_button.configure(state='normal')
            self.cancel_button.configure(state='disabled', text="Cancel")
            
            success = len(self.processed_files)
            total = len(self.current_files)
            
            self._log("\n--- Processing Complete ---")
            self._log(f"Successfully processed: {success}/{total} files")
            self.status_var.set(f"Complete! Processed {success}/{total} files")
            
            if success > 0:
                messagebox.showinfo(
                    "Processing Complete",
                    f"Successfully processed {success} of {total} PDF files."
                )
                
                if success > 1:
                    if messagebox.askyesno("Open Files", "Open all processed PDFs?"):
                        for _, out in self.processed_files:
                            open_file_cross_platform(out)
        
        def _generate_preview_async(self, pdf_file: str):
            """Generate preview in background thread."""
            def generate():
                try:
                    base = os.path.basename(pdf_file)
                    output_path = os.path.join(tempfile.gettempdir(), f"(Preview) {base}")
                    
                    self.remover.remove_watermarks(
                        pdf_file,
                        output_path,
                        status_cb=lambda msg: self._update_status(f"[Preview] {msg}")
                    )
                    
                    self.root.after(0, lambda: self._update_preview(pdf_file, output_path))
                    
                except Exception as e:
                    self._log_threadsafe(f"Preview error: {str(e)}")
            
            self._log("🔍 Generating preview...")
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
        
        def _update_preview(self, original_path: str, processed_path: Optional[str]):
            """Update preview images."""
            self.preview_original_path = original_path
            self.preview_processed_path = processed_path
            
            # Clear old images
            self._preview_images.clear()
            
            # Update original preview
            if original_path:
                img = self._render_pdf_page(original_path)
                if img:
                    self._preview_images.append(img)
                    self.original_preview.configure(image=img, text="")
                else:
                    self.original_preview.configure(image='', text="Failed to render\n(Click to open)")
            
            # Update processed preview
            if processed_path:
                img = self._render_pdf_page(processed_path)
                if img:
                    self._preview_images.append(img)
                    self.processed_preview.configure(image=img, text="")
                else:
                    self.processed_preview.configure(image='', text="Failed to render\n(Click to open)")
            else:
                self.processed_preview.configure(image='', text="Processing...")
        
        def _render_pdf_page(self, pdf_path: str, page_num: int = 0):
            """Render PDF page to PhotoImage."""
            if not HAS_PIL:
                return None
            
            try:
                if not pdf_path or not os.path.exists(pdf_path):
                    return None
                
                doc = fitz.open(pdf_path)
                if page_num >= len(doc):
                    page_num = 0
                
                page = doc.load_page(page_num)
                mat = fitz.Matrix(1.0, 1.0)  # Lower resolution for tkinter
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize to fit preview area (max 350x450)
                img.thumbnail((350, 450), Image.Resampling.LANCZOS)
                
                doc.close()
                
                # Convert to PhotoImage
                return ImageTk.PhotoImage(img)
                
            except Exception:
                return None
        
        def _clear_preview(self, text: str = "No preview available"):
            """Clear preview labels."""
            self._preview_images.clear()
            self.preview_original_path = None
            self.preview_processed_path = None
            self.original_preview.configure(image='', text=text)
            self.processed_preview.configure(image='', text=text)
        
        def _open_original(self):
            """Open original PDF."""
            if self.preview_original_path:
                open_file_cross_platform(self.preview_original_path)
        
        def _open_processed(self):
            """Open processed PDF."""
            if self.preview_processed_path:
                open_file_cross_platform(self.preview_processed_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point - uses PyQt6 if available, otherwise tkinter."""
    
    if USING_PYQT6:
        print("Using PyQt6 GUI")
        app = QApplication(sys.argv)
        app.setApplicationName("PDF Watermark Remover")
        app.setStyle('Fusion')
        
        window = WatermarkRemovalApp()
        window.show()
        
        sys.exit(app.exec())
    
    elif USING_TKINTER:
        print("PyQt6 not found, using tkinter fallback GUI")
        if not HAS_PIL:
            print("Note: Install Pillow for PDF preview support (pip install Pillow)")
        
        root = tk.Tk()
        app = TkinterWatermarkApp(root)
        root.mainloop()
    
    else:
        print("Error: No GUI framework available")
        sys.exit(1)


if __name__ == "__main__":
    main()