#!/usr/bin/env python3
"""
PDF Watermark Remover - Clean Implementation
============================================
A modular watermark detection and removal tool with PyQt6 GUI.

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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
from functools import lru_cache

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QSplitter,
    QScrollArea, QMessageBox, QFileDialog, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QFont, QIcon

import fitz  # PyMuPDF


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

# Quick check for diagonal values - matches rotation matrices where a,b are non-trivial
# For angles 15°-75°, values range from ~0.26 to ~0.97
# This catches: 0.3-0.9 range values that indicate rotation
RE_DIAGONAL_QUICK = re.compile(r'[-]?0\.[2-9]\d*\s+[-]?0\.[2-9]\d*')

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
    # Check if it's NOT close to 0, 90, 180, 270, 360
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
    element_type: str  # 'text', 'image', 'vector', etc.

    # Position info (normalized 0-1)
    norm_position: Tuple[float, float] = (0.0, 0.0)
    area_fraction: float = 0.0

    # For text elements
    text_content: str = ""
    font_name: str = ""
    font_size: float = 0.0
    direction_angle: float = 0.0
    color: int = 0

    # For matching
    signature: str = ""

    def __post_init__(self):
        """Generate signature for matching similar elements."""
        if not self.signature:
            self.signature = self._generate_signature()

    def _generate_signature(self) -> str:
        """Create a signature for matching similar watermarks across pages."""
        # Round position to allow for minor variations
        pos_bin = position_bin(self.norm_position[0], self.norm_position[1])

        # Round angle to nearest 5 degrees
        angle_bin = round(self.direction_angle / 5) * 5

        # Round area to 1 decimal
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
        """Detect watermarks in the document."""
        pass

    @abstractmethod
    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        """Remove detected watermarks. Returns count of removed elements."""
        pass

    def _get_pages_to_sample(self, doc: fitz.Document) -> List[int]:
        """Get list of page indices to sample - spread across document."""
        total = len(doc)
        if total <= self.sample_pages:
            return list(range(total))
        
        # Sample pages spread across document, not just first N
        # This catches watermarks on pages with different sizes
        sample_indices = []
        step = total / self.sample_pages
        for i in range(self.sample_pages):
            idx = int(i * step)
            sample_indices.append(min(idx, total - 1))
        
        # Ensure we don't have duplicates
        return sorted(set(sample_indices))

    def _get_hit_threshold(self, sampled_pages: int) -> int:
        """Calculate minimum hits needed to consider something a watermark."""
        return max(1, int(sampled_pages * self.repeat_threshold))


# =============================================================================
# CASE #1: DIAGONAL TEXT WATERMARK DETECTOR
# =============================================================================

class DiagonalTextDetector(WatermarkDetector):
    """
    Detects diagonal text watermarks in main content streams.
    
    Characteristics targeted:
    - Text blocks with non-horizontal/vertical rotation
    - Repeated across most pages in similar positions
    - Often large relative to page size
    - Typically center-positioned
    - Rotation applied via Tm or cm operators inside BT...ET blocks
    """

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
        """Detect diagonal text watermarks by analyzing text blocks across pages."""
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Analyzing pages for diagonal text watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        # Collect candidates from sampled pages
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

        # Find signatures that appear on enough pages
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
        """Extract potential diagonal text watermarks from a page."""
        candidates = []

        try:
            text_dict = page.get_text("dict")
        except Exception:
            return candidates

        for block in text_dict.get("blocks", []):
            # Only process text blocks
            if block.get("type") != 0:
                continue

            bbox = block.get("bbox")
            if not bbox:
                continue

            # Check area fraction
            area_frac = calculate_area_fraction(bbox, page_rect)
            if not (self.min_area_fraction <= area_frac <= self.max_area_fraction):
                continue

            # Analyze lines for direction/rotation
            max_angle = 0.0
            font_name = ""
            font_size = 0.0
            color = 0
            text_parts = []

            for line in block.get("lines", []):
                direction = line.get("dir")
                if direction:
                    angle = abs(direction_to_angle(direction))
                    # Normalize angle to 0-90 range for comparison
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

            # Check if angle is diagonal (not horizontal or vertical)
            if not (self.min_angle <= max_angle <= self.max_angle):
                continue

            # This is a diagonal text candidate
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
        """
        Check whether a BT/ET segment contains diagonal text based on cm/Tm matrices.
        """
        # Quick check first
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
        """
        Remove only BT...ET segments that contain diagonal text matrices.
        """
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
        """Remove diagonal text watermarks by editing content streams in-place.
        
        Once watermarks are detected, we scan ALL streams in the document for
        diagonal text blocks. This catches watermarks on pages with different
        sizes or orientations that might have been missed during sampling.
        """
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing diagonal text watermarks from all streams...")

        # Use precompiled module-level patterns
        bt_pattern = RE_BT_ET
        cm_pattern = RE_CM_MATRIX
        tm_pattern = RE_TM_MATRIX

        removed_count = 0

        # Scan ALL streams in the document, not just matched pages
        # This catches watermarks on pages with different sizes/orientations
        try:
            xref_count = doc.xref_length()
        except Exception:
            return 0

        for xref in range(1, xref_count):
            try:
                raw_bytes = doc.xref_stream(xref)
                if raw_bytes is None:
                    continue
                content = raw_bytes.decode("latin-1", errors="ignore")
                
                # Quick check - must have text blocks
                if 'BT' not in content or 'ET' not in content:
                    continue
                
                # Quick check - must have potential diagonal values
                if not RE_DIAGONAL_QUICK.search(content):
                    continue
                
                new_content, removed_here = self._remove_diagonal_text_segments_from_stream(
                    content, bt_pattern, cm_pattern, tm_pattern
                )

                if removed_here > 0 and new_content != content:
                    doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                    removed_count += removed_here
                    
            except Exception:
                continue

        if status_cb:
            status_cb(f"Removed {removed_count} diagonal text watermark block(s)")

        return removed_count

    def _identify_watermark_regions(self, page: fitz.Page, page_num: int,
                                    page_rect: fitz.Rect,
                                    signatures: Set[str]) -> List[fitz.Rect]:
        """Identify regions containing watermark content on a page."""
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
    """
    Detects watermarks stored as XObject Forms.
    
    Characteristics targeted:
    - Watermark content is in a separate XObject Form stream
    - Diagonal rotation can be applied in TWO ways:
      Case #2: Rotation INSIDE the XObject stream (cm/Tm operators)
      Case #3: Rotation OUTSIDE at invocation time (cm before /Name Do)
    - Same XObject referenced across multiple pages
    - Often contains text or colored overlays
    
    Structure examples:
    Case #2 (internal rotation):
      - Content stream: q 1 0 0 1 x y cm /X2 Do Q
      - XObject stream: q 0.707 0.707 -0.707 0.707 ... cm BT ... ET Q
    
    Case #3 (external rotation):
      - Content stream: q 0.819 0.574 -0.574 0.819 x y cm /Fm0 Do Q
      - XObject stream: BT 1 0 0 1 x y Tm ... ET (no rotation inside)
    """

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_angle: float = 15.0,
                 max_angle: float = 75.0):
        super().__init__(sample_pages, repeat_threshold)
        self.min_angle = min_angle
        self.max_angle = max_angle
        
        # Use precompiled module-level patterns
        self.cm_pattern = RE_CM_MATRIX
        self.tm_pattern = RE_TM_MATRIX

    def _normalize_name(self, name: Any) -> str:
        """Convert XObject name to a clean string without leading '/'."""
        if isinstance(name, bytes):
            name = name.decode("latin-1", errors="ignore")
        name = str(name)
        return name.lstrip("/")

    def _matrix_is_diagonal(self, a: float, b: float) -> bool:
        """Check if matrix coefficients represent a diagonal rotation."""
        return is_diagonal_angle(a, b, self.min_angle, self.max_angle)

    def _stream_has_diagonal(self, content: str) -> bool:
        """Return True if stream content has any diagonal cm/Tm transform (Case #2)."""
        # Quick check first
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
        """
        Find XObjects invoked with diagonal transformation (Case #3).
        
        Returns dict mapping XObject name -> xref if invoked with diagonal cm.
        """
        diagonal_invocations: Dict[str, int] = {}
        
        xrefs = page.get_contents()
        if not xrefs:
            return diagonal_invocations
            
        if isinstance(xrefs, int):
            xrefs = [xrefs]
        
        # Pattern to find: q ... cm /Name Do ... Q blocks
        # We look for cm immediately before /Name Do
        do_pattern = re.compile(
            r"q\s+[^Q]*?"  # q followed by anything (non-greedy)
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"  # a b
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+"  # c d
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm\s*"  # e f cm
            r"/(\w+)\s+Do"  # /Name Do
            r"[^Q]*?Q",  # ... Q
            re.DOTALL
        )
        
        # Also need to find xref for each XObject name
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
        """Detect XObject-based watermarks (both Case #2 and Case #3)."""
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Analyzing XObject Forms for watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        # Track which XObject xrefs appear on which pages with diagonal transform
        xref_page_count: Counter = Counter()
        xref_to_names: Dict[int, Set[str]] = defaultdict(set)
        
        # Cache: xref -> has internal diagonal (Case #2)
        xref_has_internal_diagonal: Dict[int, bool] = {}

        for page_num in pages_to_sample:
            page = doc.load_page(page_num)

            try:
                xobjects = page.get_xobjects()
            except Exception:
                continue

            # Case #3: Find XObjects invoked with diagonal cm
            diagonal_invocations = self._find_diagonal_xobject_invocations(doc, page)

            seen_on_this_page: Set[int] = set()

            for xobj in xobjects:
                if len(xobj) < 2:
                    continue

                xref = xobj[0]
                raw_name = xobj[1]
                name = self._normalize_name(raw_name)

                xref_to_names[xref].add(name)

                # Check Case #2: internal diagonal (only check once per xref)
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

                # Check Case #3: external diagonal (at invocation)
                is_external_diag = name in diagonal_invocations

                # If either case matches, count this XObject
                if (xref_has_internal_diagonal.get(xref, False) or is_external_diag):
                    if xref not in seen_on_this_page:
                        xref_page_count[xref] += 1
                        seen_on_this_page.add(xref)

        # Decide which XObject xrefs are watermarks
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

        # Store mapping for removal phase
        result.candidates_by_page = {
            "watermark_xrefs": watermark_xrefs,
            "xref_to_names": {xref: list(names) for xref, names in xref_to_names.items()}
        }

        return result

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        """
        Remove XObject watermarks.

        Strategy:
        1) Blank the XObject Form streams (so any /Name Do draws nothing).
        2) Also scrub `/Name Do` invocations from page content streams.
        """
        candidates = result.candidates_by_page or {}
        watermark_xrefs: Set[int] = set()

        if "watermark_xrefs" in candidates:
            watermark_xrefs = set(candidates["watermark_xrefs"] or [])
        else:
            # Fallback: parse from watermark_signatures as ints
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

        # ------------------------------------------------------------------
        # 1) Blank the XObject streams
        # ------------------------------------------------------------------
        removed_streams = 0

        for xref in watermark_xrefs:
            try:
                if doc.xref_stream(xref) is not None:
                    doc.update_stream(xref, b"")
                    removed_streams += 1
            except Exception:
                continue

        # ------------------------------------------------------------------
        # 2) Scrub `/Name Do` from page content streams
        # ------------------------------------------------------------------
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
    """
    Detects watermarks deeply nested inside XObject chains.
    
    Characteristics targeted:
    - Watermark is inside nested XObjects (e.g., Fm0 → Fm0 → TPL* → page)
    - Container XObjects have very large/infinite BBox (e.g., -32768 to 32768)
    - Diagonal text with low alpha/opacity
    - Common names: Fm0, Fm1, etc.
    
    Key insight: page.get_xobjects() only returns first-level XObjects.
    Nested XObjects must be found by scanning ALL document xrefs.
    """

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
        
        # Use precompiled module-level patterns for performance
        self.cm_pattern = RE_CM_MATRIX
        self.tm_pattern = RE_TM_MATRIX
        self.do_pattern = RE_DO_CALL
        self.bt_et_pattern = RE_BT_ET

    def _normalize_name(self, name: Any) -> str:
        """Convert XObject name to a clean string without leading '/'."""
        if isinstance(name, bytes):
            name = name.decode("latin-1", errors="ignore")
        name = str(name)
        return name.lstrip("/")

    def _matrix_is_diagonal(self, a: float, b: float) -> bool:
        """Check if matrix coefficients represent a diagonal rotation."""
        return is_diagonal_angle(a, b, self.min_angle, self.max_angle)

    def _has_large_bbox(self, bbox: tuple) -> bool:
        """Check if BBox is abnormally large (watermark container signature)."""
        if not bbox or len(bbox) < 4:
            return False
        try:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            return width > self.large_bbox_threshold or height > self.large_bbox_threshold
        except (TypeError, IndexError):
            return False

    def _stream_has_diagonal_text(self, content: str) -> bool:
        """Check if stream has diagonal text rendering (cm/Tm before BT...ET)."""
        # Quick check first - if no potential diagonal values, skip expensive regex
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
        """Check if stream contains text rendering."""
        # Fast string check before regex
        return 'BT' in content and 'ET' in content

    def _is_form_xobject(self, obj_str: str) -> bool:
        """Check if xref object dictionary indicates a Form XObject."""
        return '/Subtype /Form' in obj_str or '/Subtype/Form' in obj_str

    def _get_xobject_bbox(self, doc: fitz.Document, xref: int) -> Optional[tuple]:
        """Get BBox of an XObject from its dictionary."""
        try:
            xobj_dict = doc.xref_object(xref)
            bbox_match = re.search(r'/BBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', xobj_dict)
            if bbox_match:
                return (
                    float(bbox_match.group(1)),
                    float(bbox_match.group(2)),
                    float(bbox_match.group(3)),
                    float(bbox_match.group(4))
                )
        except Exception:
            pass
        return None

    def _scan_all_xrefs_for_watermarks(self, doc: fitz.Document, 
                                        status_cb: Optional[Callable] = None) -> Set[int]:
        """
        Scan ALL xrefs in the document to find watermark streams.
        
        This handles multiple patterns:
        - Form XObjects with large bbox and diagonal text
        - Small Form XObjects with diagonal text only
        - Case #4: Non-Form streams with diagonal text (watermarks embedded in content streams)
        
        This is necessary because nested watermarks don't appear in page.get_xobjects().
        """
        watermark_xrefs: Set[int] = set()
        
        try:
            xref_count = doc.xref_length()
        except Exception:
            return watermark_xrefs
        
        # Track stream size distribution for diagonal text streams
        diagonal_streams_by_size: Dict[int, List[int]] = defaultdict(list)  # size_bucket -> [xrefs]
        
        # Cache for object strings to avoid re-reading
        for xref in range(1, xref_count):
            try:
                # Get stream content first - skip if no stream
                stream = doc.xref_stream(xref)
                if not stream:
                    continue
                
                content = stream.decode("latin-1", errors="ignore")
                stream_len = len(content)
                
                # Quick checks first - avoid expensive operations
                # Must have text blocks to be a watermark
                if 'BT' not in content or 'ET' not in content:
                    continue
                
                # Check for diagonal text
                has_diagonal = self._stream_has_diagonal_text(content)
                if not has_diagonal:
                    continue
                
                # If we get here, this stream has diagonal text
                # Track by size for Case #4 pattern matching
                size_bucket = (stream_len // 10) * 10
                diagonal_streams_by_size[size_bucket].append(xref)
                
                # Check if it's a Form XObject (get object dict only if needed)
                obj_str = doc.xref_object(xref)
                is_form = '/Subtype /Form' in obj_str or '/Subtype/Form' in obj_str
                
                if is_form:
                    # Check for large bbox
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
                    
                    # Small Form streams that are purely watermark content
                    if stream_len < 2000 and 'Do' not in content:
                        watermark_xrefs.add(xref)
                        if status_cb:
                            status_cb(f"  Found candidate: xref {xref} (small Form with diagonal text)")
                
            except Exception:
                continue
        
        # Case #4: Find diagonal text streams that appear consistently
        # If many streams of same size have diagonal text, likely watermarks
        num_pages = len(doc)
        for size_bucket, xref_list in diagonal_streams_by_size.items():
            count = len(xref_list)
            
            # Check if count matches page patterns (1x, 2x, 3x pages)
            is_page_pattern = any(
                abs(count - num_pages * mult) <= num_pages * 0.1
                for mult in (1, 2, 3, 4)
            )
            
            if is_page_pattern and count >= 10:
                watermark_xrefs.update(xref_list)
                if status_cb:
                    status_cb(f"  Found Case #4 pattern: {count} streams of ~{size_bucket} bytes (diagonal text)")
        
        return watermark_xrefs

    def _find_xrefs_that_call(self, doc: fitz.Document, target_xrefs: Set[int]) -> Set[int]:
        """Find all xrefs that call the target xrefs via /Name Do."""
        callers: Set[int] = set()
        
        # First, build a mapping from xref to names
        xref_to_names: Dict[int, Set[str]] = defaultdict(set)
        
        try:
            xref_count = doc.xref_length()
            
            # Find names for target xrefs by scanning XObject resource dictionaries
            for xref in range(1, xref_count):
                try:
                    obj_str = doc.xref_object(xref)
                    # Look for /XObject << /Name X 0 R >> patterns
                    for target_xref in target_xrefs:
                        pattern = re.compile(r'/(\w+)\s+' + str(target_xref) + r'\s+0\s+R')
                        for match in pattern.finditer(obj_str):
                            xref_to_names[target_xref].add(match.group(1))
                except Exception:
                    continue
            
            # Now find streams that call these names
            target_names: Set[str] = set()
            for names in xref_to_names.values():
                target_names.update(names)
            
            if not target_names:
                return callers
            
            for xref in range(1, xref_count):
                try:
                    stream = doc.xref_stream(xref)
                    if not stream:
                        continue
                    content = stream.decode("latin-1", errors="ignore")
                    
                    for name in target_names:
                        if re.search(r'/' + re.escape(name) + r'\s+Do', content):
                            callers.add(xref)
                            break
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return callers

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        """Detect nested XObject watermarks by scanning all document xrefs."""
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning all XObjects for nested watermarks...")

        # Scan ALL xrefs to find watermark candidates
        watermark_xrefs = self._scan_all_xrefs_for_watermarks(doc, status_cb)
        
        # FALLBACK: If pattern-based detection found nothing, but PyMuPDF sees diagonal text,
        # do an aggressive scan of ALL streams with diagonal text
        if not watermark_xrefs:
            if status_cb:
                status_cb("Pattern detection found nothing, checking PyMuPDF text extraction...")
            
            # Check if PyMuPDF detects any diagonal text
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
            
            # If PyMuPDF sees diagonal text, aggressively collect ALL streams with diagonal text
            if has_diagonal_text:
                if status_cb:
                    status_cb("PyMuPDF detected diagonal text, performing aggressive scan...")
                
                watermark_xrefs = self._aggressive_diagonal_scan(doc, status_cb)
        
        if not watermark_xrefs:
            return DetectionResult(description="No nested watermarks found")

        # Build name mapping for found watermarks
        xref_to_names: Dict[int, Set[str]] = defaultdict(set)
        
        # Try to find names by checking page resources
        for page_num in range(min(5, len(doc))):
            try:
                page = doc.load_page(page_num)
                for xobj in page.get_xobjects():
                    if len(xobj) >= 2 and xobj[0] in watermark_xrefs:
                        xref_to_names[xobj[0]].add(self._normalize_name(xobj[1]))
            except Exception:
                pass
        
        # Also scan document for resource definitions
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
        """
        Aggressively scan ALL streams for diagonal text.
        Used as fallback when pattern-based detection fails.
        """
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
                
                # Check for diagonal text
                if self._stream_has_diagonal_text(content) and self._stream_has_text(content):
                    watermark_xrefs.add(xref)
                    
            except Exception:
                continue
        
        if status_cb and watermark_xrefs:
            status_cb(f"  Aggressive scan found {len(watermark_xrefs)} diagonal text streams")
        
        return watermark_xrefs

    def remove(self, doc: fitz.Document, result: DetectionResult,
               status_cb: Optional[Callable] = None) -> int:
        """
        Remove nested XObject watermarks by surgically removing diagonal text blocks.
        
        Unlike previous approach that blanked entire streams, this:
        1. Scans ALL streams in the document
        2. Finds BT...ET blocks with diagonal Tm transforms
        3. Removes only those blocks, preserving other content
        """
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

        # Use precompiled pattern
        bt_et_pattern = RE_BT_ET
        
        removed_count = 0

        # Surgically remove diagonal text blocks from identified watermark streams
        for xref in watermark_xrefs:
            try:
                stream = doc.xref_stream(xref)
                if stream is None:
                    continue
                    
                content = stream.decode("latin-1", errors="ignore")
                
                # Find and remove diagonal BT...ET blocks
                new_content, blocks_removed = self._remove_diagonal_text_blocks(content, bt_et_pattern)
                
                if blocks_removed > 0 and new_content != content:
                    doc.update_stream(xref, new_content.encode("latin-1", errors="ignore"))
                    removed_count += blocks_removed
                    
            except Exception:
                continue

        # Also scan ALL other streams for diagonal text blocks
        # This catches watermarks in page content streams and other XObjects
        if status_cb:
            status_cb("Scanning all streams for remaining diagonal text...")
        
        try:
            xref_count = doc.xref_length()
            for xref in range(1, xref_count):
                if xref in watermark_xrefs:
                    continue  # Already processed
                    
                try:
                    stream = doc.xref_stream(xref)
                    if stream is None:
                        continue
                        
                    content = stream.decode("latin-1", errors="ignore")
                    
                    # Quick checks using precompiled pattern
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
        """
        Remove BT...ET blocks that contain diagonal Tm transforms.
        Returns (new_content, blocks_removed).
        """
        removed = 0
        result_parts = []
        last_end = 0
        
        for match in bt_et_pattern.finditer(content):
            block = match.group(0)
            
            # Check if this block has a diagonal Tm
            if self._block_has_diagonal_tm(block):
                # Remove this block - append content before it
                result_parts.append(content[last_end:match.start()])
                last_end = match.end()
                removed += 1
        
        # Append remaining content
        result_parts.append(content[last_end:])
        
        return "".join(result_parts), removed

    def _block_has_diagonal_tm(self, block: str) -> bool:
        """Check if a BT...ET block contains a diagonal Tm transform."""
        for match in self.tm_pattern.finditer(block):
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                if self._matrix_is_diagonal(a, b):
                    return True
            except (ValueError, TypeError):
                continue
        
        # Also check cm patterns within the block
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

# Known watermark phrases (exact match, case-insensitive)
WATERMARK_PHRASES = {
    'confidential information - do not distribute',
}


class HorizontalTextWatermarkDetector(WatermarkDetector):
    """
    Detects horizontal text watermarks using two methods:
    
    Method A - Phrase matching:
    - Matches known watermark phrases like "CONFIDENTIAL INFORMATION - DO NOT DISTRIBUTE"
    
    Method B - Visual characteristics (for unknown text):
    - Gray color (#808080 or similar grayscale)
    - Transparency (alpha < 200)
    - Located in margins (top or bottom 10% of page)
    - Repeats on most/all pages in same position
    
    Both methods require text to repeat consistently across pages.
    """

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 custom_phrases: Optional[Set[str]] = None,
                 margin_threshold: float = 0.1):  # Top/bottom 10% of page
        super().__init__(sample_pages, repeat_threshold)
        
        # Phrase matching
        self.watermark_phrases = WATERMARK_PHRASES.copy()
        if custom_phrases:
            self.watermark_phrases.update(p.lower() for p in custom_phrases)
        
        # Visual detection thresholds
        self.margin_threshold = margin_threshold
        self.gray_threshold = 40  # Max diff between R,G,B to be "gray"
        self.alpha_threshold = 200  # Alpha below this = transparent

    def _is_gray_color(self, color_int: int) -> bool:
        """Check if color is a shade of gray."""
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        return max_diff <= self.gray_threshold

    def _is_in_margin(self, bbox: Tuple[float, float, float, float], 
                      page_height: float) -> bool:
        """Check if text is in top or bottom margin."""
        y0, y1 = bbox[1], bbox[3]
        top_margin = page_height * self.margin_threshold
        bottom_margin = page_height * (1 - self.margin_threshold)
        return y1 <= top_margin or y0 >= bottom_margin

    def _text_contains_watermark_phrase(self, text: str) -> bool:
        """Check if text contains any known watermark phrase."""
        text_lower = text.lower()
        for phrase in self.watermark_phrases:
            if phrase in text_lower:
                return True
        return False

    def _is_horizontal(self, direction: Tuple[float, float]) -> bool:
        """Check if direction vector indicates horizontal text."""
        if not direction or len(direction) < 2:
            return False
        dx, dy = abs(direction[0]), abs(direction[1])
        return dx > 0.98 and dy < 0.2

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        """Detect horizontal text watermarks."""
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning for horizontal text watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        # Track two types of signatures
        phrase_signatures: Counter = Counter()  # Method A: phrase-based
        visual_signatures: Counter = Counter()  # Method B: visual characteristics
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
                
                # Method A: Check for known phrases (anywhere on page)
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
                
                # Method B: Check visual characteristics (must be in margin)
                elif is_gray and is_transparent and self._is_in_margin(bbox, page_height):
                    # Include size in signature for precise matching
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

        # Find patterns that repeat across pages
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
        """Remove horizontal text watermarks from all pages."""
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing horizontal text watermarks...")

        candidates = result.candidates_by_page or {}
        examples = candidates.get("examples", {})
        
        # Separate reference data by detection method
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
                
                # Method A: Check for known phrases
                if self._text_contains_watermark_phrase(block_text):
                    # Verify position matches a detected pattern
                    for ref_bbox in phrase_bboxes:
                        if (abs(bbox[0] - ref_bbox[0]) < 30 and 
                            abs(bbox[1] - ref_bbox[1]) < 30):
                            should_remove = True
                            break
                
                # Method B: Check visual characteristics + position match
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
    """
    Detects watermarks rendered as vector paths instead of text.
    
    Characteristics targeted:
    - Vector drawings (lines, curves, Bézier paths)
    - Low fill opacity (typically 10-30%)
    - Often black or gray fill color
    - No stroke (filled paths only)
    - Repeats across pages in similar positions
    - These are typically glyph outlines drawn as paths
    
    Detection strategy:
    - Find all low-opacity filled drawings
    - Group by Y position (same horizontal band = same line of text)
    - Count how many such drawings appear per page
    - If similar count appears on multiple pages, it's a watermark
    
    This handles Case #6 where watermarks look like text visually
    but are rendered as vector drawings in the PDF.
    """

    def __init__(self,
                 sample_pages: int = 5,
                 repeat_threshold: float = 0.8,
                 min_opacity: float = 0.05,
                 max_opacity: float = 0.35):
        super().__init__(sample_pages, repeat_threshold)
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity

    def _is_watermark_drawing(self, drawing: dict) -> Tuple[bool, float]:
        """
        Check if a drawing has watermark characteristics.
        Returns (is_watermark, fill_opacity).
        """
        # Must have fill with low opacity
        fill_opacity = drawing.get("fill_opacity")
        if fill_opacity is None:
            return False, 0.0
        
        if not (self.min_opacity <= fill_opacity <= self.max_opacity):
            return False, 0.0
        
        # Should have a fill color
        fill = drawing.get("fill")
        if fill is None:
            return False, 0.0
        
        return True, fill_opacity

    def detect(self, doc: fitz.Document, status_cb: Optional[Callable] = None) -> DetectionResult:
        """Detect vector-based watermarks by analyzing drawings across pages."""
        if len(doc) == 0:
            return DetectionResult()

        if status_cb:
            status_cb("Scanning for vector/path-based watermarks...")

        pages_to_sample = self._get_pages_to_sample(doc)
        hit_threshold = self._get_hit_threshold(len(pages_to_sample))

        # Track watermark characteristics per page
        # Key: (opacity_bin, y_band) -> count per page
        opacity_counts_per_page: Dict[int, Counter] = {}  # page -> Counter of opacity bins
        all_watermark_drawings: Dict[int, List[dict]] = {}  # page -> list of drawings
        
        for page_num in pages_to_sample:
            page = doc.load_page(page_num)
            page_rect = page.rect

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

                # Bin by opacity (to 2 decimal places)
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

        # Find opacity bins that appear consistently across pages
        # A watermark will have similar count of drawings with same opacity on each page
        watermark_opacity_bins: Set[int] = set()
        
        if len(opacity_counts_per_page) >= hit_threshold:
            # Get all opacity bins that appear on multiple pages
            all_bins: Set[int] = set()
            for counter in opacity_counts_per_page.values():
                all_bins.update(counter.keys())
            
            for opacity_bin in all_bins:
                # Count how many pages have drawings with this opacity
                pages_with_bin = sum(
                    1 for counter in opacity_counts_per_page.values()
                    if counter[opacity_bin] > 0
                )
                
                if pages_with_bin >= hit_threshold:
                    # Check if counts are similar (within 50% variation)
                    counts = [
                        counter[opacity_bin] 
                        for counter in opacity_counts_per_page.values()
                        if counter[opacity_bin] > 0
                    ]
                    if counts:
                        avg_count = sum(counts) / len(counts)
                        # Must have at least 3 drawings (characters)
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
        """Remove vector watermarks by surgically editing content streams."""
        if not result.watermark_signatures:
            return 0

        if status_cb:
            status_cb("Removing vector/path-based watermarks (surgical removal)...")

        candidates = result.candidates_by_page or {}
        watermark_opacity_bins: Set[int] = set()
        
        # Get opacity bins from signatures
        for sig in result.watermark_signatures:
            try:
                watermark_opacity_bins.add(int(sig))
            except ValueError:
                continue
        
        # Also get from candidates if available
        if "opacity_bins" in candidates:
            watermark_opacity_bins.update(candidates["opacity_bins"])

        if not watermark_opacity_bins:
            return 0

        # Convert to actual opacity values for matching
        watermark_opacities = {b / 100.0 for b in watermark_opacity_bins}
        
        # Step 1: Find ExtGState resources with matching opacity values
        watermark_gs_names = self._find_watermark_graphics_states(doc, watermark_opacities)
        
        if status_cb:
            status_cb(f"Found {len(watermark_gs_names)} graphics states with watermark opacity")

        if not watermark_gs_names:
            return 0

        removed_count = 0

        # Step 2: Remove path operations that use these graphics states
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
                
                # Quick check - must have a graphics state call and a fill operation
                has_gs = any(f'/{gs} gs' in content or f'/{gs}\ngs' in content or f'/{gs} \ngs' in content 
                            for gs in watermark_gs_names)
                has_fill = 'f*' in content or (' f ' in content) or ('\nf\n' in content) or content.endswith(' f') or content.endswith('\nf')
                
                if not (has_gs and has_fill):
                    continue
                
                # Remove watermark path blocks
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
        """
        Find ExtGState names that have fill opacity matching target values.
        Returns set of graphics state names (e.g., {'GS1', 'gs0'}).
        """
        gs_names: Set[str] = set()
        
        try:
            xref_count = doc.xref_length()
        except Exception:
            return gs_names
        
        # Pattern to find fill opacity in ExtGState: /ca 0.18
        ca_pattern = re.compile(r'/ca\s+([\d.]+)')
        
        # First, find all ExtGState objects and their opacity values
        gs_xref_to_opacity: Dict[int, float] = {}
        
        for xref in range(1, xref_count):
            try:
                obj_str = doc.xref_object(xref)
                if '/Type /ExtGState' not in obj_str and '/Type/ExtGState' not in obj_str:
                    # Also check for ExtGState without explicit Type
                    if '/ca ' not in obj_str and '/ca\n' not in obj_str:
                        continue
                
                # Look for fill opacity
                match = ca_pattern.search(obj_str)
                if match:
                    opacity = float(match.group(1))
                    gs_xref_to_opacity[xref] = opacity
                    
            except Exception:
                continue
        
        # Find which xrefs have matching opacity
        matching_xrefs: Set[int] = set()
        for xref, opacity in gs_xref_to_opacity.items():
            for target in target_opacities:
                if abs(opacity - target) < 0.02:  # 2% tolerance
                    matching_xrefs.add(xref)
                    break
        
        if not matching_xrefs:
            return gs_names
        
        # Now find the names these are referenced by in resource dictionaries
        # Pattern: /GSname X 0 R where X is the xref
        for xref in range(1, xref_count):
            try:
                obj_str = doc.xref_object(xref)
                if '/ExtGState' not in obj_str:
                    continue
                
                # Look for references to our matching xrefs
                for gs_xref in matching_xrefs:
                    # Pattern: /Name X 0 R
                    pattern = re.compile(r'/(\w+)\s+' + str(gs_xref) + r'\s+0\s+R')
                    for match in pattern.finditer(obj_str):
                        gs_names.add(match.group(1))
                        
            except Exception:
                continue
        
        return gs_names

    def _remove_watermark_path_blocks(self, content: str, 
                                       gs_names: Set[str]) -> Tuple[str, int]:
        """
        Remove q...Q blocks that use watermark graphics states and contain fill operations.
        Returns (new_content, blocks_removed).
        """
        if not gs_names:
            return content, 0
        
        # Build pattern to match graphics state usage
        gs_pattern = '|'.join(re.escape(name) for name in gs_names)
        
        # Pattern to find q...Q blocks with watermark graphics state and fill
        # This matches: q ... /GSname gs ... f or f* ... Q
        block_pattern = re.compile(
            r'q\s+'  # q followed by whitespace
            r'(?:[^Q]*?)'  # non-greedy match of anything except Q
            r'/(' + gs_pattern + r')\s+gs'  # graphics state call
            r'(?:[^Q]*?)'  # more content
            r'(?:f\*|(?<![a-zA-Z])f(?![a-zA-Z]))'  # fill operation (f* or standalone f)
            r'(?:[^Q]*?)'  # trailing content
            r'Q',  # closing Q
            re.DOTALL
        )
        
        removed = 0
        result_parts = []
        last_end = 0
        
        for match in block_pattern.finditer(content):
            # Verify this block looks like a path operation (has path operators)
            block = match.group(0)
            
            # Must have path construction operators (m, l, c, etc.)
            has_path_ops = bool(re.search(r'(?<![a-zA-Z])[mlcvyh](?![a-zA-Z])', block))
            
            if has_path_ops:
                # Remove this block
                result_parts.append(content[last_end:match.start()])
                last_end = match.end()
                removed += 1
        
        result_parts.append(content[last_end:])
        
        return "".join(result_parts), removed

    def _combine_nearby_rects(self, rects: List[fitz.Rect], 
                               gap_threshold: float = 5) -> List[fitz.Rect]:
        """Combine overlapping or touching rectangles into larger regions."""
        if not rects:
            return []

        # Sort by y then x
        sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
        
        combined = []
        current = sorted_rects[0]

        for rect in sorted_rects[1:]:
            # Check if this rect overlaps or is very close to current
            if (rect.y0 <= current.y1 + gap_threshold and
                rect.x0 <= current.x1 + gap_threshold and
                rect.y1 >= current.y0 - gap_threshold):
                # Merge
                current = fitz.Rect(
                    min(current.x0, rect.x0),
                    min(current.y0, rect.y0),
                    max(current.x1, rect.x1),
                    max(current.y1, rect.y1)
                )
            else:
                combined.append(current)
                current = rect

        combined.append(current)
        return combined


# =============================================================================
# WATERMARK REMOVER - MAIN CLASS
# =============================================================================

class WatermarkRemover:
    """Main class that orchestrates watermark detection and removal."""

    def __init__(self):
        self.detectors: List[WatermarkDetector] = []
        self._setup_default_detectors()

    def _setup_default_detectors(self):
        """Initialize default watermark detectors."""
        # Case #1: Diagonal text watermarks (in main content stream)
        self.detectors.append(DiagonalTextDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0,
            min_area_fraction=0.02,
            max_area_fraction=0.5
        ))
        
        # Case #2 & #3: XObject-based watermarks (separate form objects)
        # Handles both internal rotation (Case #2) and external rotation (Case #3)
        self.detectors.append(XObjectWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0
        ))
        
        # Case #4: Nested XObject watermarks (deeply nested in XObject chains)
        self.detectors.append(NestedXObjectWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_angle=15.0,
            max_angle=75.0,
            large_bbox_threshold=10000.0  # Detect containers with huge bbox
        ))
        
        # Case #5: Horizontal text watermarks (CONFIDENTIAL, DO NOT DISTRIBUTE, etc.)
        self.detectors.append(HorizontalTextWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7
        ))
        
        # Case #6: Vector/path-based watermarks (text drawn as paths)
        self.detectors.append(VectorWatermarkDetector(
            sample_pages=5,
            repeat_threshold=0.7,
            min_opacity=0.05,
            max_opacity=0.35
        ))

    def remove_watermarks(self, input_path: str, output_path: str,
                          status_cb: Optional[Callable] = None) -> int:
        """
        Main entry point for watermark removal.

        Args:
            input_path: Path to input PDF
            output_path: Path for output PDF
            status_cb: Optional callback for status updates

        Returns:
            Total count of removed watermark elements
        """
        try:
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            # Run each detector
            for detector in self.detectors:
                detector_name = detector.__class__.__name__
                if status_cb:
                    status_cb(f"Running {detector_name}...")

                # Detect watermarks
                result = detector.detect(doc, status_cb)

                if result.watermark_signatures:
                    if status_cb:
                        status_cb(result.description)

                    # Remove detected watermarks
                    removed = detector.remove(doc, result, status_cb)
                    total_removed += removed

            # Save output
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()

            if status_cb:
                status_cb(f"Complete: removed {total_removed} watermark elements")

            return total_removed

        except Exception as e:
            raise RuntimeError(f"Watermark removal failed: {str(e)}")


# =============================================================================
# PROCESSING THREADS
# =============================================================================

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


# =============================================================================
# GUI WIDGETS
# =============================================================================

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
        """Render a PDF page to QPixmap."""
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


# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================

class WatermarkRemovalApp(QMainWindow):
    """Main application window."""

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

        # Title
        title = QLabel("PDF Watermark Removal Tool")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_files_dropped)
        left_layout.addWidget(self.drop_zone)

        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        # Buttons
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

        # Status
        self.status_label = QLabel("Ready to process PDF files")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        left_layout.addWidget(self.status_label)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(240)
        self.log_text.setPlaceholderText("Processing log will appear here...")
        left_layout.addWidget(self.log_text)

        splitter.addWidget(left_panel)

        # Right panel - Preview
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

        # Generate preview for single file
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
# MAIN ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PDF Watermark Remover")
    app.setStyle('Fusion')

    window = WatermarkRemovalApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()