#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Watermark Remover — Lite (PyMuPDF-only)
- GUI implemented with tkinter (stdlib), so the only external dependency is PyMuPDF.
- Keeps the original app's core GUI concepts:
  • file picker (multi-select)
  • start / cancel buttons with progress bar
  • status + log pane
  • two-pane preview (original vs processed) of page 1
  • checkboxes: generic text removal, vector/object removal
- Processing & preview run on worker threads; UI remains responsive.

+ Added:
  • "Non-XObject Nuke" button that outputs "(Only XObjects) <file>.pdf"
  • "Nuke Preview" checkbox (off by default). When ON, preview shows the XObjects-only result.
  • Nuke logic matches the provided 'strip_non_xobject_rendering' behavior: remove BT..ET, BI..EI, sh paints, and path paint ops, preserving only Do()
"""

import os
import sys
import re
import math
import hashlib
import tempfile
import threading
import queue
import platform
from pathlib import Path
from collections import defaultdict, Counter

import fitz  # PyMuPDF

# ---- tkinter UI (stdlib only) ----
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import base64

APP_TITLE = "PDF Watermark Remover — Lite (PyMuPDF-only)"
VERSION = "1.1.0"


def open_file_cross_platform(path: str):
    try:
        if not path or not os.path.exists(path):
            return
        system = platform.system()
        if system == "Darwin":
            os.spawnlp(os.P_NOWAIT, "open", "open", path)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            os.spawnlp(os.P_NOWAIT, "xdg-open", "xdg-open", path)
    except Exception:
        pass


# ---------------------- UTIL ----------------------

def _rect_center_norm(rect: fitz.Rect, page_rect: fitz.Rect):
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    return ((cx - page_rect.x0) / page_rect.width, (cy - page_rect.y0) / page_rect.height)


def _rect_area_frac(rect: fitz.Rect, page_rect: fitz.Rect):
    return max(0.0, min(1.0, rect.get_area() / max(1.0, page_rect.get_area())))


def _merge_close_rects(rects, max_gap=6):
    if not rects:
        return []
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            r = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                s = rects[j]
                rr = fitz.Rect(r.x0 - max_gap, r.y0 - max_gap, r.x1 + max_gap, r.y1 + max_gap)
                if rr.intersects(s):
                    r = r | s
                    used[j] = True
                    changed = True
            used[i] = True
            out.append(r)
        rects = out
    return rects


# ---------------------- Watermark Remover Core ----------------------

class WatermarkRemover:
    """Static helper for watermark removal—used by both preview and processing workers."""

    # ---------- NEW: Non-XObject Nuke helpers (keep only Do) ----------
    BT_ET_RE   = re.compile(rb"\bBT\b.*?\bET\b", re.S)                # remove text blocks
    INLINE_IMG = re.compile(rb"\bBI\b.*?\bEI\b", re.S)                # remove inline images
    SHADE_RE   = re.compile(rb"/[^\s/]+\s+sh\b")                      # remove sh shading paints
    PAINT_RE   = re.compile(rb"\b(S|s|f\*?|F|B\*?|b\*?|n)\b")         # remove path paint ops

    @staticmethod
    def _strip_non_xobject_rendering(stream_bytes: bytes) -> bytes:
        """
        Remove all rendering that is NOT via XObject 'Do':
          - delete all text blocks (BT ... ET)
          - delete inline images (BI ... EI)
          - delete shading paints ('/Name sh')
          - delete path painting operators (stroke/fill/clip-paint) so paths don't render
        KEEP graphics state (q/Q/gs), transforms (cm), and XObject 'Do' calls.
        """
        s = stream_bytes
        s = WatermarkRemover.BT_ET_RE.sub(b"", s)
        s = WatermarkRemover.INLINE_IMG.sub(b"", s)
        s = WatermarkRemover.SHADE_RE.sub(b"", s)
        s = WatermarkRemover.PAINT_RE.sub(b"", s)
        return s

    @staticmethod
    def nuke_non_xobjects(input_path: str, output_path: str, status_cb=None) -> int:
        """
        Create a copy of the PDF where only XObject draws (Do) remain by stripping
        non-XObject rendering ops in-place. Matches the reference logic.
        """
        if status_cb:
            status_cb("Non-XObject Nuke: opening document…")
        doc = fitz.open(input_path)
        if doc.is_encrypted:
            raise RuntimeError("PDF is password-protected")

        changed_pages = 0
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            xrefs = page.get_contents()
            if not xrefs:
                continue
            if not isinstance(xrefs, (list, tuple)):
                xrefs = [xrefs]
            changed = False
            for cx in xrefs:
                try:
                    data = doc.xref_stream(cx) or b""
                except Exception:
                    data = b""
                if not data:
                    continue
                new_data = WatermarkRemover._strip_non_xobject_rendering(data)
                if new_data != data:
                    try:
                        doc.update_stream(cx, new_data)
                        changed = True
                    except Exception:
                        pass
            if changed:
                try:
                    page.clean_contents()
                except Exception:
                    pass
                changed_pages += 1

        if status_cb:
            status_cb(f"Non-XObject Nuke: updated {changed_pages} page(s).")
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        return changed_pages

    @staticmethod
    def remove_watermarks(input_path, output_path,
                          remove_generic=True,
                          handle_vector=True,
                          cfg=None,
                          status_cb=None):
        """
        Main pipeline:
          1) Vector/object repeated overlays (XObject hashing + q..Q signatures)
          2) Generic text passes (repeating phrases + diagonal big text)
          3) Artifact-tagged watermarks
          4) Fallback (ONLY IF needed): geometry-based text/vector removal, then visual repeat-mask redaction as last resort
        """
        try:
            if cfg is None:
                cfg = {}
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            # 1) Vector/object watermark handler (repeated overlays across pages)
            if handle_vector:
                if status_cb:
                    status_cb("Scanning for repeated vector/object watermarks…")
                total_removed += WatermarkRemover._remove_vector_object_watermarks(
                    doc,
                    sample_pages=cfg.get("sample_pages", 5),
                    min_vector_ops_sample=cfg.get("min_vector_ops_sample", 5),
                    min_vector_ops_remove=cfg.get("min_vector_ops_remove", 5),
                    repeat_threshold=cfg.get("repeat_threshold", 0.8),
                )

            # 2) Generic repeating + diagonal big text
            if remove_generic:
                if status_cb:
                    status_cb("Detecting & removing generic repeating text watermarks…")
                total_removed += WatermarkRemover._remove_generic_watermarks(
                    doc,
                    diag_min_deg=cfg.get("diag_min_deg", 10),
                    diag_max_deg=cfg.get("diag_max_deg", 90),
                    min_font_pt=cfg.get("min_font_pt", 9)
                )

            # 3) Artifact-marked content
            if status_cb:
                status_cb("Cleaning artifact-based watermark blocks…")
            total_removed += WatermarkRemover._remove_artifact_watermarks(doc)

            # 4) Safer, geometry-first fallback (auto) if nothing changed
            if total_removed == 0:
                if status_cb:
                    status_cb("No effect detected — trying geometry-based fallback…")
                removed_geom = WatermarkRemover._fallback_geometry_based(
                    doc,
                    sample_pages=cfg.get("sample_pages", 5),
                    repeat_threshold=cfg.get("repeat_threshold", 0.8),
                    diag_min_deg=cfg.get("diag_min_deg", 10),
                    diag_max_deg=cfg.get("diag_max_deg", 90),
                    min_font_pt=cfg.get("min_font_pt", 9),
                    status_cb=status_cb
                )
                total_removed += removed_geom

                if removed_geom == 0:
                    if status_cb:
                        status_cb("Trying last-resort: visual repeat mask (targeted redaction)…")
                    total_removed += WatermarkRemover._fallback_visual_repeat_mask(
                        doc,
                        sample_pages=cfg.get("sample_pages", 5),
                        grid=cfg.get("visual_grid", 64),
                        gray_range_tol=cfg.get("visual_stability_tol", 3),
                        max_total_area_frac=cfg.get("visual_max_area_frac", 0.05),
                        status_cb=status_cb
                    )

            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return total_removed
        except Exception:
            raise

    # ---------- VECTOR / OBJECT-BASED WATERMARK (XObject hashing + q..Q majority) ----------
    @staticmethod
    def _remove_vector_object_watermarks(
        doc,
        sample_pages=5,
        min_vector_ops_sample=5,
        min_vector_ops_remove=5,
        repeat_threshold=0.8,
    ):
        if len(doc) == 0:
            return 0

        # --- helpers / regex ---
        qQ_split = re.compile(r"(q.*?Q)", re.S)
        text_ops_re = re.compile(r"BT|Tj|TJ", re.I)
        path_ops_re = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|cm)(?![A-Za-z])")
        xobj_do_full = re.compile(r"/([A-Za-z0-9_.#-]+)\s+Do")
        opseq_re = re.compile(r"(?<![A-Za-z])(?:q|Q|cm|m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|Do|gs|sh|SCN?|scn?)(?![A-Za-z])")
        num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

        sample_n = min(sample_pages, len(doc))
        hit_min = max(1, int(round(sample_n * repeat_threshold)))

        def _sig(chunk: str) -> str:
            opseq = " ".join(opseq_re.findall(chunk))
            norm = num_re.sub("<n>", chunk)
            norm = re.sub(r"\s+", " ", norm)
            return hashlib.sha1((opseq + "|" + norm).encode("utf-8", "ignore")).hexdigest()

        # ---------- Stage A: find repeated XObjects by content hash ----------
        per_page_xobj_hashes = []

        def _xobject_stream_hash(doc, xref):
            try:
                data = doc.xref_stream(xref) or b""
            except Exception:
                data = b""
            return hashlib.sha1(b"%d|" % len(data) + data[:4096] + b"|" + data[-4096:]).hexdigest()

        for pno in range(sample_n):
            page = doc.load_page(pno)
            page_hashes = set()
            try:
                raw = page.read_contents().decode("latin-1", errors="ignore")
            except Exception:
                per_page_xobj_hashes.append(page_hashes)
                continue

            # Robust mapping name -> xref
            name_to_xref = {}
            try:
                xobjs = page.get_xobjects()
                for it in xobjs:
                    if isinstance(it, dict):
                        nm = it.get("name")
                        xr = it.get("xref")
                        if nm is None or xr is None:
                            continue
                        if isinstance(nm, bytes):
                            nm = nm.decode("latin-1", "ignore")
                        name_to_xref[str(nm).lstrip("/")] = xr
                    else:
                        nm = None
                        xr = None
                        for v in it:
                            if isinstance(v, int) and xr is None:
                                xr = v
                            if isinstance(v, (bytes, str)):
                                s = v.decode("latin-1", "ignore") if isinstance(v, bytes) else v
                                if s.startswith("/") and nm is None:
                                    nm = s
                        if nm and xr:
                            name_to_xref[nm.lstrip("/")] = xr
            except Exception:
                pass

            used_names = set(xobj_do_full.findall(raw))
            for nm in used_names:
                xr = name_to_xref.get(nm)
                if not xr:
                    continue
                try:
                    s = (doc.xref_stream(xr) or b"").decode("latin-1", "ignore")
                except Exception:
                    s = ""
                if not s:
                    continue
                if text_ops_re.search(s):  # skip text forms; handled by text logic
                    continue
                if len(path_ops_re.findall(s)) < min_vector_ops_sample:
                    continue

                h = _xobject_stream_hash(doc, xr)
                page_hashes.add((xr, h))
            per_page_xobj_hashes.append(page_hashes)

        hash_count = {}
        xref_by_hash = {}
        for ph in per_page_xobj_hashes:
            for xr, h in ph:
                hash_count[h] = hash_count.get(h, 0) + 1
                xref_by_hash.setdefault(h, xr)

        repeated_hashes = {h for h, c in hash_count.items() if c >= hit_min}

        removed = 0

        # Blank repeated XObject streams once (affects all placements)
        for h in repeated_hashes:
            xr = xref_by_hash.get(h)
            if not xr:
                continue
            try:
                doc.update_stream(xr, b"")
                removed += 1
            except Exception:
                pass

        # ---------- Stage B: q..Q repeated-chunk remover (majority threshold) ----------
        per_page_sigs = []
        for pno in range(sample_n):
            try:
                page = doc.load_page(pno)
                raw = page.read_contents().decode("latin-1", errors="ignore")
                parts = qQ_split.split(raw)
            except Exception:
                per_page_sigs.append(set())
                continue

            page_s = set()
            for i, chunk in enumerate(parts):
                if i % 2 == 1:  # q..Q block
                    vector_ops = len(path_ops_re.findall(chunk))
                    if vector_ops < min_vector_ops_sample:
                        continue
                    if text_ops_re.search(chunk):
                        continue
                    page_s.add(_sig(chunk))
            per_page_sigs.append(page_s)

        if per_page_sigs:
            sig_count = {}
            for s in per_page_sigs:
                for sig in s:
                    sig_count[sig] = sig_count.get(sig, 0) + 1
            repeated_sigs = {s for s, c in sig_count.items() if c >= hit_min}
        else:
            repeated_sigs = set()

        if repeated_sigs:
            for pno in range(len(doc)):
                page = doc.load_page(pno)
                xrefs = page.get_contents()
                if not xrefs:
                    continue
                try:
                    raw = page.read_contents().decode("latin-1", errors="ignore")
                except Exception:
                    continue
                parts = qQ_split.split(raw)
                changed = False
                out = []
                for i, chunk in enumerate(parts):
                    if i % 2 == 1:
                        sig = _sig(chunk)
                        if sig in repeated_sigs:
                            if min_vector_ops_remove > 0:
                                vops = len(path_ops_re.findall(chunk))
                                if vops < min_vector_ops_remove:
                                    out.append(chunk)
                                    continue
                            removed += 1
                            changed = True
                            continue
                    out.append(chunk)
                if changed:
                    new_raw = "".join(out).encode("latin-1", errors="ignore")
                    try:
                        if isinstance(xrefs, (list, tuple)) and xrefs:
                            page.parent.update_stream(xrefs[0], new_raw)
                            for xr in xrefs[1:]:
                                page.parent.update_stream(xr, b"")
                        else:
                            page.parent.update_stream(xrefs, new_raw)
                    except Exception:
                        pass

        return removed

    # ---------- Text / generic passes ----------
    @staticmethod
    def _remove_generic_watermarks(doc, diag_min_deg=10, diag_max_deg=90, min_font_pt=10):
        if len(doc) < 2:
            return 0
        removed = 0
        cands = WatermarkRemover._detect_repeating_text_patterns(doc)
        for page in doc:
            removed += WatermarkRemover._safe_delete_matching_annots(
                page, lambda a: WatermarkRemover._annot_matches_watermark(a, None)
            )
        for page in doc:
            if cands:
                removed += WatermarkRemover._strip_text_objects(page, phrases=cands)
            removed += WatermarkRemover._strip_diagonal_large_text(
                page, min_pt=min_font_pt, min_deg=diag_min_deg, max_deg=diag_max_deg
            )
        return removed

    @staticmethod
    def _detect_repeating_text_patterns(doc):
        sample_pages = min(5, len(doc))
        text_positions = {}
        for page_num in range(sample_pages):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if len(text) < 5 or text.lower() in ["the", "and", "or", "of", "to", "in", "for"]:
                                continue
                            bbox = span["bbox"]
                            rel_x = (bbox[0] + bbox[2]) / 2 / page.rect.width
                            rel_y = (bbox[1] + bbox[3]) / 2 / page.rect.height
                            key = f"{rel_x:.2f},{rel_y:.2f}"
                            text_positions.setdefault(text, {})
                            text_positions[text][key] = text_positions[text].get(key, 0) + 1
        cands = []
        for text, positions in text_positions.items():
            if max(positions.values()) >= min(3, int(sample_pages * 0.6)):
                cands.append(text)
        return cands

    @staticmethod
    def _strip_text_objects(page, phrases):
        try:
            if not page.get_contents():
                return 0
            up_phrases = [p.upper() for p in phrases if p and p.strip()]
            if not up_phrases:
                return 0

            raw = page.read_contents().decode('latin-1', errors='ignore')
            artifact_pattern = re.compile(r"/Artifact\b[^B]*?/Watermark\b.*?BDC(.*?)EMC", re.S | re.I)
            raw2, n_art = artifact_pattern.subn("", raw)

            parts = re.split(r"(BT.*?ET)", raw2, flags=re.S | re.I)
            changed = False
            out_parts = []
            removed_blocks = 0

            tj_simple = re.compile(r"\((?:\\.|[^()])*\)\s*Tj", re.S)
            tj_array = re.compile(r"\[\s*(?:\((?:\\.|[^()])*\)\s*-?\d+\s*)+\]\s*TJ", re.S)

            def block_has_phrase(block_text):
                strings = []
                for m in tj_simple.finditer(block_text):
                    strings.append(m.group(0))
                for m in tj_array.finditer(block_text):
                    strings.append(m.group(0))

                if not strings:
                    return False

                def extract_literals(s):
                    lits = re.findall(r"\((?:\\.|[^()])*\)", s, flags=re.S)
                    out = []
                    for lit in lits:
                        t = lit[1:-1]
                        t = t.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
                        out.append(t)
                    return " ".join(out)

                merged = " ".join(extract_literals(s) for s in strings).upper()
                merged = re.sub(r"\s+", " ", merged)
                return any(p in merged for p in up_phrases)

            for i, chunk in enumerate(parts):
                if i % 2 == 1:
                    if block_has_phrase(chunk):
                        removed_blocks += 1
                        changed = True
                        continue
                out_parts.append(chunk)

            if changed or n_art:
                new_stream = "".join(out_parts).encode('latin-1', errors='ignore')
                xrefs = page.get_contents()
                if isinstance(xrefs, (list, tuple)) and xrefs:
                    page.parent.update_stream(xrefs[0], new_stream)
                    for x in xrefs[1:]:
                        page.parent.update_stream(x, b"")
                else:
                    page.parent.update_stream(xrefs, new_stream)

            return removed_blocks + n_art
        except Exception:
            return 0

    @staticmethod
    def _strip_diagonal_large_text(page, min_pt=10, min_deg=10, max_deg=90):
        removed = 0
        try:
            text = page.get_text("dict")
            if not text or "blocks" not in text:
                return 0
            has_candidates = False
            for block in text.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    d = line.get("dir")
                    if not d:
                        continue
                    angle = abs(math.degrees(math.atan2(d[1], d[0])))
                    if not (min_deg <= angle <= max_deg):
                        continue
                    if any(span.get("size", 0) >= min_pt for span in line.get("spans", [])):
                        has_candidates = True
                        break
                if has_candidates:
                    break
            if not has_candidates:
                return 0

            raw = page.read_contents().decode('latin-1', errors='ignore')
            parts = re.split(r"(BT.*?ET)", raw, flags=re.S | re.I)
            changed = False
            out_parts = []

            tm_rot = re.compile(
                r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm",
                re.I
            )
            tf_big = re.compile(r"/[A-Za-z0-9]+\s+(\d{2,})\s+Tf", re.I)

            for i, chunk in enumerate(parts):
                if i % 2 == 1:
                    has_big = any(float(s) >= min_pt for s in tf_big.findall(chunk))
                    rot_hit = False
                    for a, b, c, d, _, _ in tm_rot.findall(chunk):
                        try:
                            a, b, c, d = float(a), float(b), float(c), float(d)
                            ang = abs(math.degrees(math.atan2(b, a)))
                            if min_deg <= ang <= max_deg:
                                rot_hit = True
                                break
                        except Exception:
                            pass
                    if has_big and rot_hit:
                        removed += 1
                        changed = True
                        continue
                out_parts.append(chunk)

            if changed:
                new_stream = "".join(out_parts).encode('latin-1', errors='ignore')
                xrefs = page.get_contents()
                if isinstance(xrefs, (list, tuple)) and xrefs:
                    page.parent.update_stream(xrefs[0], new_stream)
                    for x in xrefs[1:]:
                        page.parent.update_stream(x, b"")
                else:
                    page.parent.update_stream(xrefs, new_stream)
            return removed
        except Exception:
            return 0

    @staticmethod
    def _safe_delete_matching_annots(page, predicate):
        removed = 0
        try:
            annot = page.first_annot
        except Exception:
            annot = None
        while annot:
            try:
                next_annot = annot.next
            except Exception:
                next_annot = None
            try:
                if predicate(annot):
                    try:
                        page.delete_annot(annot)
                        removed += 1
                    except Exception:
                        pass
            except Exception:
                pass
            annot = next_annot
        return removed

    @staticmethod
    def _annot_matches_watermark(annot, phrases_upper=None):
        try:
            t = (annot.type[1] or "").upper()
        except Exception:
            t = ""
        try:
            info = annot.info or {}
        except Exception:
            info = {}
        contents = str(info.get("content", "") or "").upper()
        name = str(info.get("name", "") or "").upper()
        if t in {"STAMP", "WATERMARK"}:
            return True
        if phrases_upper:
            for p in phrases_upper:
                if p in contents or p in name:
                    return True
        return False

    @staticmethod
    def _remove_artifact_watermarks(doc):
        removed_count = 0
        for page in doc:
            page.clean_contents()
            xref = page.get_contents()[0] if page.get_contents() else None
            if xref:
                try:
                    content_lines = page.read_contents().splitlines()
                    modified_lines = []
                    skip_until_emc = False
                    for line in content_lines:
                        line_str = line.decode('latin-1', errors='ignore')
                        if '/Artifact' in line_str and '/Watermark' in line_str:
                            skip_until_emc = True
                            removed_count += 1
                            continue
                        if skip_until_emc and 'EMC' in line_str:
                            skip_until_emc = False
                            continue
                        if not skip_until_emc:
                            modified_lines.append(line)
                    if len(modified_lines) != len(content_lines):
                        page.parent.update_stream(xref, b'\n'.join(modified_lines))
                except Exception:
                    pass
        return removed_count

    # ---------- Fallback: Geometry-first (auto) ----------
    @staticmethod
    def _fallback_geometry_based(doc, sample_pages=5, repeat_threshold=0.8,
                                 diag_min_deg=10, diag_max_deg=90, min_font_pt=10,
                                 status_cb=None):
        if len(doc) == 0:
            return 0

        removed = 0
        pages_to_sample = min(sample_pages, len(doc))

        # --- TEXT GEOMETRY (large + diagonal + central-ish) ---
        diag_boxes_by_page = []
        for pno in range(pages_to_sample):
            page = doc.load_page(pno)
            page_rect = page.rect
            raw = page.get_text("rawdict")
            candidates = []
            for b in raw.get("blocks", []):
                if b.get("type") != 0:  # text only
                    continue
                for l in b.get("lines", []):
                    d = l.get("dir")
                    if not d:
                        continue
                    angle = abs(math.degrees(math.atan2(d[1], d[0])))
                    if not (diag_min_deg <= angle <= diag_max_deg):
                        continue
                    size_max = 0
                    r = None
                    for s in l.get("spans", []):
                        size_max = max(size_max, float(s.get("size", 0)))
                        sb = s.get("bbox")
                        if sb:
                            r = fitz.Rect(sb) if r is None else (fitz.Rect(sb) | r)
                    if size_max >= min_font_pt and r is not None:
                        cx, cy = _rect_center_norm(r, page_rect)
                        if 0.15 <= cx <= 0.85 and 0.15 <= cy <= 0.85:
                            candidates.append(r)
            diag_boxes_by_page.append(_merge_close_rects(candidates))

        bin_hits = Counter()
        for pno in range(pages_to_sample):
            page = doc.load_page(pno)
            pr = page.rect
            for r in diag_boxes_by_page[pno]:
                cx, cy = _rect_center_norm(r, pr)
                bx = int(round(cx * 10))
                by = int(round(cy * 10))
                bin_hits[(bx, by)] += 1

        hit_min = max(1, int(round(pages_to_sample * repeat_threshold)))
        repeated_bins = {k for k, c in bin_hits.items() if c >= hit_min}

        if repeated_bins:
            if status_cb:
                status_cb("Fallback: removing repeated diagonal text by geometry…")
            tm_rot = re.compile(
                r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm",
                re.I
            )
            tf_big = re.compile(r"/[A-Za-z0-9]+\s+(\d{2,})\s+Tf", re.I)
            bt_et_split_re = re.compile(r"(BT.*?ET)", re.S | re.I)

            for pno in range(len(doc)):
                page = doc.load_page(pno)
                pr = page.rect
                xrefs = page.get_contents()
                if not xrefs:
                    continue
                try:
                    raw = page.read_contents().decode("latin-1", errors="ignore")
                except Exception:
                    continue
                parts = bt_et_split_re.split(raw)
                changed = False
                out = []
                for i, chunk in enumerate(parts):
                    remove_this = False
                    if i % 2 == 1:
                        has_big = any(float(s) >= min_font_pt for s in tf_big.findall(chunk))
                        if has_big:
                            block_rect = None
                            for a, b, c, d, e, f in tm_rot.findall(chunk):
                                try:
                                    a, b, c, d, e, f = map(float, (a, b, c, d, e, f))
                                    x0 = e - 20
                                    y0 = f - 20
                                    x1 = e + 20
                                    y1 = f + 20
                                    block_rect = fitz.Rect(x0, y0, x1, y1)
                                    break
                                except Exception:
                                    pass
                            if block_rect:
                                cx, cy = _rect_center_norm(block_rect, pr)
                                bx = int(round(cx * 10))
                                by = int(round(cy * 10))
                                if (bx, by) in repeated_bins:
                                    rot_hit = False
                                    for a, b, _, _, _, _ in tm_rot.findall(chunk):
                                        try:
                                            a, b = float(a), float(b)
                                            ang = abs(math.degrees(math.atan2(b, a)))
                                            if diag_min_deg <= ang <= diag_max_deg:
                                                rot_hit = True
                                                break
                                        except Exception:
                                            pass
                                    if rot_hit:
                                        remove_this = True
                    if remove_this:
                        removed += 1
                        changed = True
                    else:
                        out.append(chunk)

                bt_blocks = max(1, len(parts) // 2)
                hits = (len(parts) // 2) - (len(out) // 2)
                if changed and (hits / bt_blocks) <= 0.3:
                    new_raw = "".join(out).encode("latin-1", errors="ignore")
                    try:
                        if isinstance(xrefs, (list, tuple)) and xrefs:
                            page.parent.update_stream(xrefs[0], new_raw)
                            for xr in xrefs[1:]:
                                page.parent.update_stream(xr, b"")
                        else:
                            page.parent.update_stream(xrefs, new_raw)
                    except Exception:
                        pass

        # --- VECTOR GEOMETRY (large-area outlines) ---
        area_bins = Counter()
        for pno in range(pages_to_sample):
            page = doc.load_page(pno)
            pr = page.rect
            drs = []
            try:
                drs = page.get_drawings()
            except Exception:
                pass
            union = None
            for d in drs:
                r = d.get("rect")
                if not r:
                    continue
                rr = fitz.Rect(r)
                union = rr if union is None else (union | rr)
            if union and _rect_area_frac(union, pr) >= 0.05:
                cx, cy = _rect_center_norm(union, pr)
                bx = int(round(cx * 10))
                by = int(round(cy * 10))
                area_bins[(bx, by)] += 1

        hit_min = max(1, int(round(pages_to_sample * repeat_threshold)))
        repeated_area_bins = {k for k, c in area_bins.items() if c >= hit_min}

        if repeated_area_bins:
            qQ_split = re.compile(r"(q.*?Q)", re.S)
            path_ops_re = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|cm)(?![A-Za-z])")
            cm_re = re.compile(
                r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm"
            )
            for pno in range(len(doc)):
                page = doc.load_page(pno)
                pr = page.rect
                xrefs = page.get_contents()
                if not xrefs:
                    continue
                try:
                    raw = page.read_contents().decode("latin-1", errors="ignore")
                except Exception:
                    continue
                parts = qQ_split.split(raw)
                changed = False
                out = []
                for i, chunk in enumerate(parts):
                    remove_this = False
                    if i % 2 == 1:
                        vops = len(path_ops_re.findall(chunk))
                        if vops >= 5:
                            block_rect = None
                            for a, b, c, d, e, f in cm_re.findall(chunk):
                                try:
                                    a, b, c, d, e, f = map(float, (a, b, c, d, e, f))
                                    x0 = e - 40
                                    y0 = f - 40
                                    x1 = e + 40
                                    y1 = f + 40
                                    block_rect = fitz.Rect(x0, y0, x1, y1)
                                except Exception:
                                    pass
                            if block_rect:
                                cx, cy = _rect_center_norm(block_rect, pr)
                                bx = int(round(cx * 10))
                                by = int(round(cy * 10))
                                if (bx, by) in repeated_area_bins:
                                    has_diag = False
                                    for a, b, _, _, _, _ in cm_re.findall(chunk):
                                        try:
                                            a, b = float(a), float(b)
                                            ang = abs(math.degrees(math.atan2(b, a)))
                                            if 20 <= ang <= 80:
                                                has_diag = True
                                                break
                                        except Exception:
                                            pass
                                    if has_diag:
                                        remove_this = True
                    if remove_this:
                        removed += 1
                        changed = True
                    else:
                        out.append(chunk)

                qblocks = max(1, len(parts) // 2)
                hits = (len(parts) // 2) - (len(out) // 2)
                if changed and (hits / qblocks) <= 0.3:
                    new_raw = "".join(out).encode("latin-1", errors="ignore")
                    try:
                        if isinstance(xrefs, (list, tuple)) and xrefs:
                            page.parent.update_stream(xrefs[0], new_raw)
                            for xr in xrefs[1:]:
                                page.parent.update_stream(xr, b"")
                        else:
                            page.parent.update_stream(xrefs, new_raw)
                    except Exception:
                        pass

        if status_cb:
            status_cb(f"Geometry fallback removed {removed} blocks.")
        return removed

    # ---------- Fallback: Visual repeat mask (last resort; auto) ----------
    @staticmethod
    def _fallback_visual_repeat_mask(doc, sample_pages=5, grid=64, gray_range_tol=3,
                                     max_total_area_frac=0.05, status_cb=None):
        if len(doc) == 0:
            return 0

        pages_to_sample = min(sample_pages, len(doc))
        rasters = []
        widths = []
        heights = []
        zoom = 0.75  # low DPI scale

        for pno in range(sample_pages):
            try:
                page = doc.load_page(pno)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
                widths.append(pix.width)
                heights.append(pix.height)
                rasters.append(pix)
            except Exception:
                pass

        if not rasters:
            return 0

        W = min(widths)
        H = min(heights)
        tw = max(1, W // grid)
        th = max(1, H // grid)

        tile_vals = defaultdict(list)
        for pix in rasters:
            if pix.width != W or pix.height != H:
                tmp = fitz.Pixmap(fitz.csGRAY, W, H)
                tmp.clear_with(255)
                tmp.copy(pix, 0, 0)
                pix = tmp
            buf = pix.samples
            for j in range(grid):
                for i in range(grid):
                    x0 = i * tw
                    y0 = j * th
                    x1 = min(W, x0 + tw)
                    y1 = min(H, y0 + th)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    s = 0
                    n = 0
                    for yy in range(y0, y1):
                        row = yy * pix.stride
                        s += sum(buf[row + x0: row + x1])
                        n += (x1 - x0)
                    avg = s / max(1, n)
                    tile_vals[(i, j)].append(avg)

        stable_tiles = set()
        need_hits = max(2, int(round(pages_to_sample * 0.90)))
        for key, vals in tile_vals.items():
            if len(vals) < need_hits:
                continue
            rng = max(vals) - min(vals)
            if rng <= gray_range_tol:  # configurable stability tolerance
                stable_tiles.add(key)

        if not stable_tiles:
            return 0

        page0 = doc.load_page(0)
        pr = page0.rect
        scale_x = pr.width / W
        scale_y = pr.height / H

        rects = []
        for j in range(grid):
            i = 0
            while i < grid:
                if (i, j) in stable_tiles:
                    start_i = i
                    while i < grid and (i, j) in stable_tiles:
                        i += 1
                    end_i = i - 1
                    x0 = start_i * tw * scale_x + pr.x0
                    x1 = min(W, (end_i + 1) * tw) * scale_x + pr.x0
                    y0 = j * th * scale_y + pr.y0
                    y1 = min(H, (j + 1) * th) * scale_y + pr.y0
                    rects.append(fitz.Rect(x0, y0, x1, y1))
                i += 1

        rects = _merge_close_rects(rects, max_gap=8)
        rects = [r for r in rects if 0.001 <= _rect_area_frac(r, pr) <= 0.12]

        if not rects:
            return 0

        total_area_frac = sum(_rect_area_frac(r, pr) for r in rects)
        if total_area_frac > max_total_area_frac:
            return 0

        removed = 0
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            for r in rects:
                try:
                    page.add_redact_annot(r, fill=(1, 1, 1))
                except Exception:
                    pass
            try:
                page.apply_redactions()
                removed += 1
            except Exception:
                pass

        if status_cb:
            status_cb(f"Visual fallback applied on {removed} pages.")
        return removed


# ---------------------- Worker threads ----------------------

class ProcessorWorker(threading.Thread):
    def __init__(self, files, remove_generic, handle_vector, cfg, ui_queue, mode='normal'):
        super().__init__(daemon=True)
        self.files = files
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cfg = cfg.copy() if cfg else {}
        self.ui_queue = ui_queue
        self._cancel = False
        self.processed = []
        self.mode = mode  # 'normal' or 'nuke'

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self.files) if self.files else 0
        for i, f in enumerate(self.files):
            if self._cancel:
                break
            try:
                base = Path(f)
                if self.mode == 'normal':
                    out = base.parent / f"(No Watermarks) {base.name}"

                    def _status(msg):
                        self.ui_queue.put(("status", f"Processing {base.name}: {msg}"))

                    WatermarkRemover.remove_watermarks(
                        str(base), str(out),
                        remove_generic=self.remove_generic,
                        handle_vector=self.handle_vector,
                        cfg=self.cfg,
                        status_cb=_status
                    )
                else:
                    out = base.parent / f"(Only XObjects) {base.name}"

                    def _status(msg):
                        self.ui_queue.put(("status", f"Non-XObject Nuke {base.name}: {msg}"))

                    WatermarkRemover.nuke_non_xobjects(str(base), str(out), status_cb=_status)

                self.processed.append((str(base), str(out)))
                self.ui_queue.put(("log", f"✅ Processed: {base.name}\n    Output: {out.name}"))
                self.ui_queue.put(("preview", (str(base), str(out)) if i == 0 else None))
            except Exception as e:
                self.ui_queue.put(("log", f"❌ Error processing {os.path.basename(f)}: {e}"))
            pct = int(((i + 1) / max(1, total)) * 100)
            self.ui_queue.put(("progress", pct))

        self.ui_queue.put(("done", self.processed))


class PreviewWorker(threading.Thread):
    def __init__(self, file_path, remove_generic, handle_vector, cfg, ui_queue, mode='normal'):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cfg = cfg.copy() if cfg else {}
        self.ui_queue = ui_queue
        self._cancel = False
        self.tmp_output = None
        self.mode = mode  # 'normal' or 'nuke'

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            base = os.path.basename(self.file_path)
            if self.mode == 'normal':
                self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")

                def _status(msg):
                    self.ui_queue.put(("status", f"[Preview] {msg}"))

                WatermarkRemover.remove_watermarks(
                    self.file_path, self.tmp_output,
                    remove_generic=self.remove_generic,
                    handle_vector=self.handle_vector,
                    cfg=self.cfg,
                    status_cb=_status
                )
            else:
                self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - Only XObjects) {base}")

                def _status(msg):
                    self.ui_queue.put(("status", f"[Preview] {msg}"))

                WatermarkRemover.nuke_non_xobjects(
                    self.file_path, self.tmp_output, status_cb=_status
                )

            if not self._cancel:
                self.ui_queue.put(("preview", (self.file_path, self.tmp_output)))
        except Exception as e:
            if not self._cancel:
                self.ui_queue.put(("log", f"❌ Preview error: {e}"))


# ---------------------- GUI ----------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x760")
        try:
            self.iconbitmap(default="")  # no external icon by default
        except Exception:
            pass

        # config
        self.cfg = {
            "sample_pages": 5,
            "repeat_threshold": 0.8,
            "min_vector_ops_sample": 5,
            "min_vector_ops_remove": 5,
            "diag_min_deg": 10,
            "diag_max_deg": 90,
            "min_font_pt": 9,
            "visual_grid": 64,
            "visual_stability_tol": 3,
            "visual_max_area_frac": 0.05,
        }

        self.files = []
        self.processed = []
        self.proc_worker = None
        self.prev_worker = None
        self.ui_queue = queue.Queue()

        self._build_ui()
        self._poll_ui_queue()

    # ---- UI construction ----
    def _build_ui(self):
        # Title
        title = ttk.Label(self, text="PDF Watermark Removal Tool", anchor="center", font=("Segoe UI", 18, "bold"))
        title.pack(side="top", fill="x", pady=(12, 6))

        # Settings row
        settings = ttk.LabelFrame(self, text="Watermark Removal Settings")
        settings.pack(side="top", fill="x", padx=16, pady=8)
        self.var_generic = tk.BooleanVar(value=True)
        self.var_vector = tk.BooleanVar(value=True)
        self.var_nuke_preview = tk.BooleanVar(value=False)
        chk1 = ttk.Checkbutton(settings, text="Auto-detect and remove repeating text watermarks", variable=self.var_generic, command=self._on_any_setting_changed)
        chk2 = ttk.Checkbutton(settings, text="Handle vector/object-based watermarks (path overlays)", variable=self.var_vector, command=self._on_any_setting_changed)
        chk3 = ttk.Checkbutton(settings, text="Nuke Preview (show XObjects-only)", variable=self.var_nuke_preview, command=self._on_any_setting_changed)
        chk1.pack(side="left", padx=8, pady=8)
        chk2.pack(side="left", padx=8, pady=8)
        chk3.pack(side="left", padx=8, pady=8)

        # Splitter (use paned window)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=16, pady=8)

        # Left side
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        # File chooser (acts like the previous drop zone; drag&drop needs tkdnd which is not stdlib)
        file_box = ttk.LabelFrame(left, text="Files")
        file_box.pack(fill="x", pady=(0, 8))
        self.file_label = ttk.Label(file_box, text="No files selected")
        self.file_label.pack(side="left", padx=8, pady=8)
        ttk.Button(file_box, text="Choose PDFs…", command=self.choose_files).pack(side="right", padx=8, pady=8)

        # Progress
        prog_box = ttk.LabelFrame(left, text="Processing Progress")
        prog_box.pack(fill="x", pady=8)
        self.progress = ttk.Progressbar(prog_box, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=10, pady=10)
        btns = ttk.Frame(prog_box)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        self.btn_start = ttk.Button(btns, text="Start Processing", command=self.start_processing, state="disabled")
        self.btn_nuke  = ttk.Button(btns, text="Non-XObject Nuke", command=self.start_nuke_processing, state="disabled")
        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self.cancel_processing, state="disabled")
        self.btn_start.pack(side="left")
        self.btn_nuke.pack(side="left", padx=8)
        self.btn_cancel.pack(side="left", padx=8)

        # Status + log
        self.status = ttk.Label(left, text="Ready to process PDF files", foreground="#1e7e34")
        self.status.pack(fill="x", pady=(4, 2), padx=4)
        self.log = ScrolledText(left, height=12)
        self.log.pack(fill="both", expand=False, pady=(0, 8), padx=4)

        # Right side (preview)
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        ttk.Label(right, text="PDF Preview", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        preview_box = ttk.Frame(right)
        preview_box.pack(fill="both", expand=True)

        # Use tk.Labels for images
        self.orig_img_label = tk.Label(preview_box, bg="#f9f9f9", relief="groove", width=64, height=24)
        self.proc_img_label = tk.Label(preview_box, bg="#f9f9f9", relief="groove", width=64, height=24)

        # Grid layout
        preview_box.columnconfigure(0, weight=1)
        preview_box.columnconfigure(1, weight=1)
        ttk.Label(preview_box, text="Original page 1").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        ttk.Label(preview_box, text="Processed (Preview) page 1").grid(row=0, column=1, sticky="w", padx=4, pady=(0, 4))
        self.orig_img_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=4)
        self.proc_img_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=4)

    # ---- UI events ----

    def _on_any_setting_changed(self, *_):
        self._clear_preview("Refreshing preview…")
        if self.prev_worker and self.prev_worker.is_alive():
            self.prev_worker.cancel()
        if len(self.files) == 1:
            self._start_preview(self.files[0])
        else:
            self._set_preview_disabled("Preview disabled when multiple PDFs are selected.")

    def choose_files(self):
        files = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF Files", "*.pdf")])
        if not files:
            return
        self.files = list(files)
        self.processed = []
        self.log.delete("1.0", "end")

        count = len(files)
        self.file_label.config(text=f"{count} file(s) selected")
        self.btn_start.config(state="normal")
        self.btn_nuke.config(state="normal")
        self.status.config(text=f"Ready to process {count} PDF file(s)")

        if count == 1:
            first_file = files[0]
            self._update_preview(first_file, None)
            self._start_preview(first_file)
        else:
            self._set_preview_disabled("Preview disabled when multiple PDFs are selected.")

    # ---- Preview helpers ----
    def _start_preview(self, pdf_file):
        mode = 'nuke' if self.var_nuke_preview.get() else 'normal'
        self.prev_worker = PreviewWorker(
            pdf_file,
            remove_generic=self.var_generic.get(),
            handle_vector=self.var_vector.get(),
            cfg=self.cfg,
            ui_queue=self.ui_queue,
            mode=mode
        )
        self.log.insert("end", ("🧨 Generating XObjects-only preview…" if mode == 'nuke' else "🔍 Generating processed preview…") + "\n")
        self.prev_worker.start()

    def _set_preview_disabled(self, reason_text):
        self._set_photo(self.orig_img_label, None, text=reason_text)
        self._set_photo(self.proc_img_label, None, text=reason_text)

    def _clear_preview(self, text="Refreshing preview…"):
        self._set_preview_disabled(text)

    def _update_preview(self, original_path, processed_path):
        # Render page 1 of original and processed (if available)
        self._render_to_label(self.orig_img_label, original_path, page_num=0, fallback="Original preview")
        if processed_path:
            self._render_to_label(self.proc_img_label, processed_path, page_num=0, fallback="Processed preview")
        else:
            self._set_photo(self.proc_img_label, None, text="Processed (Preview) will appear here")

    def _render_to_label(self, label: tk.Label, pdf_path, page_num=0, fallback="Preview"):
        try:
            if not pdf_path or not os.path.exists(pdf_path):
                self._set_photo(label, None, text=f"{fallback} unavailable")
                return
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                page_num = 0
            page = doc.load_page(page_num)
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            data = pix.tobytes("png")
            doc.close()

            # Show PNG bytes via base64 on a tkinter PhotoImage (no PIL required)
            b64 = base64.b64encode(data).decode("ascii")
            img = tk.PhotoImage(data=b64)
            # keep reference to avoid GC
            label._img = img
            label.configure(image=img, text="")
        except Exception:
            self._set_photo(label, None, text=f"{fallback} unavailable")

    def _set_photo(self, label: tk.Label, img, text=""):
        label._img = img
        label.configure(image=img)
        if text:
            label.configure(text=text, anchor="center")
        else:
            label.configure(text="")

    # ---- Processing ----
    def start_processing(self):
        if not self.files:
            return
        self._start_processing_with_mode('normal')

    def start_nuke_processing(self):
        if not self.files:
            return
        self._start_processing_with_mode('nuke')

    def _start_processing_with_mode(self, mode='normal'):
        self.btn_start.config(state="disabled")
        self.btn_nuke.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress.config(value=0)

        self.proc_worker = ProcessorWorker(
            self.files,
            remove_generic=self.var_generic.get(),
            handle_vector=self.var_vector.get(),
            cfg=self.cfg,
            ui_queue=self.ui_queue,
            mode=mode
        )
        self.log.insert("end", ("\n--- Processing Started (Non-XObject Nuke) ---\n" if mode == 'nuke' else "\n--- Processing Started ---\n"))
        self.proc_worker.start()

    def cancel_processing(self):
        if self.proc_worker and self.proc_worker.is_alive():
            self.proc_worker.cancel()
            self.btn_cancel.config(state="disabled")
            self.status.config(text="Cancelling…")

    # ---- UI queue polling ----
    def _poll_ui_queue(self):
        try:
            while True:
                msg, payload = self.ui_queue.get_nowait()
                if msg == "status":
                    self.status.config(text=str(payload))
                elif msg == "log":
                    self.log.insert("end", str(payload) + "\n")
                    self.log.see("end")
                elif msg == "preview":
                    if payload:
                        orig, proc = payload
                        self._update_preview(orig, proc)
                elif msg == "progress":
                    self.progress.config(value=int(payload))
                elif msg == "done":
                    processed = payload or []
                    self.btn_start.config(state="normal")
                    self.btn_nuke.config(state="normal")
                    self.btn_cancel.config(state="disabled")
                    self.progress.config(value=100)
                    self.status.config(text=f"Complete! Processed {len(processed)}/{len(self.files)} files")
                    if processed:
                        self.log.insert("end", "\nOutput files saved:\n")
                        for _, outp in processed:
                            self.log.insert("end", f"  • {os.path.basename(outp)}\n")
                        self.log.insert("end", "\n")
                        self.log.see("end")
                        if len(processed) > 1:
                            if messagebox.askyesno("Open All Processed PDFs", "Open all processed PDFs now?"):
                                for _, outp in processed:
                                    open_file_cross_platform(outp)
                self.ui_queue.task_done()
        except queue.Empty:
            pass
        # schedule next poll
        self.after(66, self._poll_ui_queue)


def main():
    root = App()
    root.call("tk", "scaling", 1.25)  # HiDPI-ish default
    root.mainloop()


if __name__ == "__main__":
    main()
