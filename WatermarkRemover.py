# Professional PDF Watermark Removal Application - Simplified Version 
# (Non-destructive + Safe Annot Deletion + Smart Preview + Open-All-When-Done)
# Vector/object watermarks: samples first N pages for identical overlays (XObject hashing + q..Q majority)
# Geometry-aware fallback for "flattened" PDFs:
#   - Repeated large-diagonal TEXT regions by geometry (rawdict)
#   - Repeated large-area VECTOR overlays by drawings()
#   - Visual repeat-mask (low-res) as last resort → targeted redaction rectangles only

import sys
import os
import math
import re
import subprocess
import platform
import hashlib
from pathlib import Path
import tempfile
from collections import defaultdict, Counter
from copy import deepcopy

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QSplitter,
    QScrollArea, QMessageBox, QFileDialog,
    QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QFont, QIcon

import fitz  # PyMuPDF


def open_file_cross_platform(path: str):
    try:
        if not path or not os.path.exists(path):
            return
        system = platform.system()
        if system == "Darwin":
            subprocess.call(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


# ---------------------- UTIL ----------------------

def _rect_center_norm(rect: fitz.Rect, page_rect: fitz.Rect):
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    return ( (cx - page_rect.x0) / page_rect.width, (cy - page_rect.y0) / page_rect.height )

def _rect_area_frac(rect: fitz.Rect, page_rect: fitz.Rect):
    return max(0.0, min(1.0, rect.get_area() / max(1.0, page_rect.get_area())))

def _merge_close_rects(rects, max_gap=6):
    if not rects: return []
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False]*len(rects)
        for i in range(len(rects)):
            if used[i]: continue
            r = rects[i]
            for j in range(i+1, len(rects)):
                if used[j]: continue
                s = rects[j]
                rr = fitz.Rect(r.x0-max_gap, r.y0-max_gap, r.x1+max_gap, r.y1+max_gap)
                if rr.intersects(s):
                    r = r | s
                    used[j] = True
                    changed = True
            used[i] = True
            out.append(r)
        rects = out
    return rects


class WatermarkRemover:
    """Static helper for watermark removal—used by both preview and processing threads."""

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
            if cfg is None: cfg = {}
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            # 1) Vector/object watermark handler (repeated overlays across pages)
            if handle_vector:
                if status_cb: status_cb("Scanning for repeated vector/object watermarks…")
                total_removed += WatermarkRemover._remove_vector_object_watermarks(
                    doc,
                    sample_pages=cfg.get("sample_pages", 5),
                    min_vector_ops_sample=cfg.get("min_vector_ops_sample", 5),
                    min_vector_ops_remove=cfg.get("min_vector_ops_remove", 5),
                    repeat_threshold=cfg.get("repeat_threshold", 0.8),
                )

            # 2) Generic repeating + diagonal big text
            if remove_generic:
                if status_cb: status_cb("Detecting & removing generic repeating text watermarks…")
                total_removed += WatermarkRemover._remove_generic_watermarks(
                    doc,
                    diag_min_deg=cfg.get("diag_min_deg", 10),
                    diag_max_deg=cfg.get("diag_max_deg", 90),
                    min_font_pt=cfg.get("min_font_pt", 9)
                )

            # 3) Artifact-marked content
            if status_cb: status_cb("Cleaning artifact-based watermark blocks…")
            total_removed += WatermarkRemover._remove_artifact_watermarks(doc)

            # 4) Safer, geometry-first fallback (auto) if nothing changed
            if total_removed == 0:
                if status_cb: status_cb("No effect detected — attempting geometry-based fallback…")
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

                # If still nothing and we strongly suspect a baked-in repeat, try a visual mask
                if removed_geom == 0:
                    if status_cb: status_cb("Trying last-resort: visual repeat mask (targeted redaction)…")
                    total_removed += WatermarkRemover._fallback_visual_repeat_mask(
                        doc,
                        sample_pages=cfg.get("sample_pages", 5),
                        grid=cfg.get("visual_grid", 64),
                        gray_range_tol=cfg.get("visual_stability_tol", 5),
                        max_total_area_frac=cfg.get("visual_max_area_frac", 0.05),
                        status_cb=status_cb
                    )

            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return total_removed
        except Exception:
            raise

    # ---------- NEW: Non-XObject Nuke (keep only XObjects; delete everything else) ----------
    # Implemented to match user's reference "strip_non_xobject_rendering" behavior.
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
    def nuke_non_xobjects(input_path, output_path, status_cb=None):
        """
        Create a copy of the PDF where only XObject draws (Do) remain by stripping
        non-XObject rendering ops in-place. This matches the provided reference logic.
        """
        if status_cb: status_cb("Non-XObject Nuke: opening document…")
        doc = fitz.open(input_path)
        if doc.is_encrypted:
            raise RuntimeError("PDF is password-protected")

        kept_pages = 0
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            xrefs = page.get_contents()
            if not xrefs:
                continue
            changed = False
            for cx in (xrefs if isinstance(xrefs, (list, tuple)) else [xrefs]):
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
                kept_pages += 1

        if status_cb: status_cb(f"Non-XObject Nuke: updated {kept_pages} page(s).")
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        return kept_pages

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
        qQ_split      = re.compile(r"(q.*?Q)", re.S)
        text_ops_re   = re.compile(r"BT|Tj|TJ", re.I)
        path_ops_re   = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|cm)(?![A-Za-z])")
        xobj_do_full  = re.compile(r"/([A-Za-z0-9_.#-]+)\s+Do")
        opseq_re      = re.compile(r"(?<![A-Za-z])(?:q|Q|cm|m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|Do|gs|sh|SCN?|scn?)(?![A-Za-z])")
        num_re        = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

        sample_n = min(sample_pages, len(doc))
        hit_min  = max(1, int(round(sample_n * repeat_threshold)))

        def _sig(chunk: str) -> str:
            opseq = " ".join(opseq_re.findall(chunk))
            norm  = num_re.sub("<n>", chunk)
            norm  = re.sub(r"\s+", " ", norm)
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
                per_page_xobj_hashes.append(page_hashes); continue

            # Robust mapping name -> xref
            name_to_xref = {}
            try:
                xobjs = page.get_xobjects()
                for it in xobjs:
                    if isinstance(it, dict):
                        nm = it.get("name"); xr = it.get("xref")
                        if nm is None or xr is None: 
                            continue
                        if isinstance(nm, bytes): nm = nm.decode("latin-1", "ignore")
                        name_to_xref[str(nm).lstrip("/")] = xr
                    else:
                        nm = None; xr = None
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
                per_page_sigs.append(set()); continue

            page_s = set()
            for i, chunk in enumerate(parts):
                if i % 2 == 1:  # q..Q block
                    vector_ops = len(path_ops_re.findall(chunk))
                    if vector_ops < min_vector_ops_sample:
                        continue
                    if re.search(r"BT|Tj|TJ", chunk, re.I):
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
                                vops = len(re.findall(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\\*?|B\\*?|b\\*?|W\\*?|W|n|cm)(?![A-Za-z])", chunk))
                                if vops < min_vector_ops_remove:
                                    out.append(chunk); continue
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
                            if len(text) < 5 or text.lower() in ["the","and","or","of","to","in","for"]:
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
            if not page.get_contents(): return 0
            up_phrases = [p.upper() for p in phrases if p and p.strip()]
            if not up_phrases: return 0

            raw = page.read_contents().decode('latin-1', errors='ignore')
            artifact_pattern = re.compile(r"/Artifact\b[^B]*?/Watermark\b.*?BDC(.*?)EMC", re.S | re.I)
            raw2, n_art = artifact_pattern.subn("", raw)

            parts = re.split(r"(BT.*?ET)", raw2, flags=re.S | re.I)
            changed = False
            out_parts = []
            removed_blocks = 0

            tj_simple = re.compile(r"\((?:\\.|[^()])*\)\s*Tj", re.S)
            tj_array  = re.compile(r"\[\s*(?:\((?:\\.|[^()])*\)\s*-?\d+\s*)+\]\s*TJ", re.S)

            def block_has_phrase(block_text):
                strings = []
                for m in tj_simple.finditer(block_text): strings.append(m.group(0))
                for m in tj_array.finditer(block_text):  strings.append(m.group(0))
                if not strings: return False
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
                if block.get("type") != 0: continue
                for line in block.get("lines", []):
                    d = line.get("dir")
                    if not d: continue
                    angle = abs(math.degrees(math.atan2(d[1], d[0])))
                    if not (min_deg <= angle <= max_deg): continue
                    if any(span.get("size", 0) >= min_pt for span in line.get("spans", [])):
                        has_candidates = True; break
                if has_candidates: break
            if not has_candidates: return 0

            raw = page.read_contents().decode('latin-1', errors='ignore')
            parts = re.split(r"(BT.*?ET)", raw, flags=re.S | re.I)
            changed = False; out_parts = []

            tm_rot = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm", re.I)
            tf_big = re.compile(r"/[A-Za-z0-9]+\s+(\d{2,})\s+Tf", re.I)

            for i, chunk in enumerate(parts):
                if i % 2 == 1:
                    has_big = any(float(s) >= min_pt for s in tf_big.findall(chunk))
                    rot_hit = False
                    for a,b,c,d,_,_ in tm_rot.findall(chunk):
                        try:
                            a,b,c,d = float(a), float(b), float(c), float(d)
                            ang = abs(math.degrees(math.atan2(b, a)))
                            if min_deg <= ang <= max_deg:
                                rot_hit = True; break
                        except:
                            pass
                    if has_big and rot_hit:
                        removed += 1; changed = True; continue
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
                bx = int(round(cx * 10)); by = int(round(cy * 10))
                bin_hits[(bx, by)] += 1

        hit_min = max(1, int(round(pages_to_sample * repeat_threshold)))
        repeated_bins = {k for k, c in bin_hits.items() if c >= hit_min}

        if repeated_bins:
            if status_cb: status_cb("Fallback: removing repeated diagonal text by geometry…")
            tm_rot = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm", re.I)
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
                            for a,b,c,d,e,f in tm_rot.findall(chunk):
                                try:
                                    a,b,c,d,e,f = map(float, (a,b,c,d,e,f))
                                    x0 = e - 20; y0 = f - 20; x1 = e + 20; y1 = f + 20
                                    block_rect = fitz.Rect(x0,y0,x1,y1)
                                    break
                                except Exception:
                                    pass
                            if block_rect:
                                cx, cy = _rect_center_norm(block_rect, pr)
                                bx = int(round(cx * 10)); by = int(round(cy * 10))
                                if (bx,by) in repeated_bins:
                                    rot_hit = False
                                    for a,b,_,_,_,_ in tm_rot.findall(chunk):
                                        try:
                                            a,b = float(a), float(b)
                                            ang = abs(math.degrees(math.atan2(b, a)))
                                            if diag_min_deg <= ang <= diag_max_deg:
                                                rot_hit = True; break
                                        except:
                                            pass
                                    if rot_hit:
                                        remove_this = True
                    if remove_this:
                        removed += 1
                        changed = True
                    else:
                        out.append(chunk)

                bt_blocks = max(1, len(parts)//2)
                hits = (len(parts)//2) - (len(out)//2)
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
                bx = int(round(cx * 10)); by = int(round(cy * 10))
                area_bins[(bx,by)] += 1

        repeated_area_bins = {k for k,c in area_bins.items() if c >= hit_min}

        if repeated_area_bins:
            qQ_split = re.compile(r"(q.*?Q)", re.S)
            path_ops_re = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?|W\*?|W|n|cm)(?![A-Za-z])")
            cm_re = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm")
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
                            for a,b,c,d,e,f in cm_re.findall(chunk):
                                try:
                                    a,b,c,d,e,f = map(float, (a,b,c,d,e,f))
                                    x0 = e - 40; y0 = f - 40; x1 = e + 40; y1 = f + 40
                                    block_rect = fitz.Rect(x0,y0,x1,y1)
                                except:
                                    pass
                            if block_rect:
                                cx, cy = _rect_center_norm(block_rect, pr)
                                bx = int(round(cx*10)); by = int(round(cy*10))
                                if (bx,by) in repeated_area_bins:
                                    has_diag = False
                                    for a,b,_,_,_,_ in cm_re.findall(chunk):
                                        try:
                                            a,b = float(a), float(b)
                                            ang = abs(math.degrees(math.atan2(b, a)))
                                            if 20 <= ang <= 80:
                                                has_diag = True; break
                                        except:
                                            pass
                                    if has_diag:
                                        remove_this = True
                    if remove_this:
                        removed += 1
                        changed = True
                    else:
                        out.append(chunk)

                qblocks = max(1, len(parts)//2)
                hits = (len(parts)//2) - (len(out)//2)
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

        for pno in range(pages_to_sample):
            try:
                page = doc.load_page(pno)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
                widths.append(pix.width); heights.append(pix.height)
                rasters.append(pix)
            except Exception:
                pass

        if not rasters:
            return 0

        W = min(widths); H = min(heights)
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
                    x0 = i*tw; y0 = j*th
                    x1 = min(W, x0+tw); y1 = min(H, y0+th)
                    if x1<=x0 or y1<=y0: 
                        continue
                    s = 0; n = 0
                    for yy in range(y0, y1):
                        row = yy*pix.stride
                        s += sum(buf[row+x0: row+x1])
                        n += (x1-x0)
                    avg = s / max(1, n)
                    tile_vals[(i,j)].append(avg)

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
                if (i,j) in stable_tiles:
                    start_i = i
                    while i < grid and (i,j) in stable_tiles:
                        i += 1
                    end_i = i-1
                    x0 = start_i*tw*scale_x + pr.x0
                    x1 = min(W, (end_i+1)*tw)*scale_x + pr.x0
                    y0 = j*th*scale_y + pr.y0
                    y1 = min(H, (j+1)*th)*scale_y + pr.y0
                    rects.append(fitz.Rect(x0,y0,x1,y1))
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
                    page.add_redact_annot(r, fill=(1,1,1))
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


# ---------------------- THREADS ----------------------

class PDFProcessorThread(QThread):
    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(list)
    preview_ready = pyqtSignal(str, str)

    def __init__(self, pdf_files,
                 remove_generic=True,
                 handle_vector=True,
                 cfg=None,
                 mode='normal'):
        super().__init__()
        self.pdf_files = pdf_files
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cfg = deepcopy(cfg) if cfg else {}
        self.cancel_requested = False
        self.processed_files = []
        self.mode = mode  # 'normal' or 'nuke'

    def run(self):
        total_files = len(self.pdf_files)
        for i, pdf_file in enumerate(self.pdf_files):
            if self.cancel_requested:
                break
            try:
                if self.mode == 'normal':
                    self.status_updated.emit(f"Processing {os.path.basename(pdf_file)}...")
                    input_path = Path(pdf_file)
                    output_path = input_path.parent / f"(No Watermarks) {input_path.name}"
                    def _status(msg): self.status_updated.emit(msg)
                    WatermarkRemover.remove_watermarks(
                        str(input_path), str(output_path),
                        remove_generic=self.remove_generic,
                        handle_vector=self.handle_vector,
                        cfg=self.cfg,
                        status_cb=_status
                    )
                else:
                    self.status_updated.emit(f"Non-XObject Nuke: {os.path.basename(pdf_file)}...")
                    input_path = Path(pdf_file)
                    output_path = input_path.parent / f"(No Watermarks) {input_path.name}"
                    def _status(msg): self.status_updated.emit(msg)
                    WatermarkRemover.nuke_non_xobjects(str(input_path), str(output_path), status_cb=_status)

                self.file_processed.emit(pdf_file, str(output_path))
                self.processed_files.append((pdf_file, str(output_path)))
                if i == 0:
                    self.preview_ready.emit(pdf_file, str(output_path))

                progress = int(((i + 1) / total_files) * 100)
                self.progress_updated.emit(progress)
            except Exception as e:
                self.error_occurred.emit(pdf_file, f"Processing error: {str(e)}")
        self.processing_complete.emit(self.processed_files)

    def cancel(self):
        self.cancel_requested = True


class PreviewGeneratorThread(QThread):
    preview_ready = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)

    def __init__(self, pdf_file,
                 remove_generic=True,
                 handle_vector=True,
                 cfg=None,
                 mode='normal'):
        super().__init__()
        self.pdf_file = pdf_file
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cfg = deepcopy(cfg) if cfg else {}
        self.cancel_requested = False
        self.tmp_output = None
        self.mode = mode  # 'normal' or 'nuke'

    def run(self):
        try:
            base = os.path.basename(self.pdf_file)
            if self.mode == 'normal':
                self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")
                def _status(msg): self.status_updated.emit(f"[Preview] {msg}")
                WatermarkRemover.remove_watermarks(
                    self.pdf_file, self.tmp_output,
                    remove_generic=self.remove_generic,
                    handle_vector=self.handle_vector,
                    cfg=self.cfg,
                    status_cb=_status
                )
            else:
                self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - Only XObjects) {base}")
                def _status(msg): self.status_updated.emit(f"[Preview] {msg}")
                WatermarkRemover.nuke_non_xobjects(self.pdf_file, self.tmp_output, status_cb=_status)

            if not self.cancel_requested:
                self.preview_ready.emit(self.pdf_file, self.tmp_output)
        except Exception as e:
            if not self.cancel_requested:
                self.error_occurred.emit(self.pdf_file, f"Preview error: {e}")

    def cancel(self):
        self.cancel_requested = True


# ---------------------- UI WIDGETS ----------------------

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PDFPreviewWidget(QScrollArea):
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
        self.original_label.clicked.connect(lambda: open_file_cross_platform(self.current_pdfs[0]))
        self.processed_label.clicked.connect(lambda: open_file_cross_platform(self.current_pdfs[1]))

    def set_preview_disabled(self, reason_text: str):
        self.current_pdfs = (None, None)
        self.original_label.setPixmap(QPixmap()); self.original_label.setText(reason_text)
        self.processed_label.setPixmap(QPixmap()); self.processed_label.setText(reason_text)

    def clear_preview(self, text="Refreshing preview…"):
        self.set_preview_disabled(text)

    def update_preview(self, original_path, processed_path):
        self.current_pdfs = (original_path, processed_path)
        try:
            op = self._render_pdf_page(original_path, 0)
            pp = self._render_pdf_page(processed_path, 0)
            if op:
                self.original_label.setPixmap(op.scaled(
                    400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            else:
                self.original_label.setPixmap(QPixmap()); self.original_label.setText("Original PDF will appear here")
            if pp:
                self.processed_label.setPixmap(pp.scaled(
                    400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            else:
                self.processed_label.setPixmap(QPixmap()); self.processed_label.setText("Processed PDF will appear here")
        except Exception as e:
            print(f"Preview error: {e}")

    def _render_pdf_page(self, pdf_path, page_num=0):
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
            pm = QPixmap(); pm.loadFromData(data)
            doc.close()
            return pm
        except Exception:
            return None


class DropZoneWidget(QLabel):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(150)
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
        self.setText("📁 Drag and drop PDF files here\n\nOr click to browse...")
        self.mousePressEvent = self.open_file_dialog

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            pdf_files = [url.toLocalFile() for url in event.mimeData().urls()
                         if url.toLocalFile().lower().endswith('.pdf')]
            if pdf_files:
                event.acceptProposedAction()
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
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
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

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            pdf_files = [url.toLocalFile() for url in event.mimeData().urls()
                         if url.toLocalFile().lower().endswith('.pdf')]
            if pdf_files:
                self.files_dropped.emit(pdf_files)
                event.acceptProposedAction()
        self.dragLeaveEvent(None)

    def open_file_dialog(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Files", "", "PDF Files (*.pdf)")
        if files:
            self.files_dropped.emit(files)


# ---------------------- MAIN APP ----------------------

class WatermarkRemovalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor_thread = None
        self.preview_thread = None
        self.processed_files = []
        self.cfg = self._default_cfg()

        self.init_ui()
        self.setWindowTitle("PDF Watermark Remover by Emil Ferdman (With Help from ChatGPT)")
        self.setGeometry(100, 100, 1200, 760)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass

    # ---- configuration ----
    def _default_cfg(self):
        return {
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

    def init_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget); layout.setSpacing(15); layout.setContentsMargins(20,20,20,20)

        title_label = QLabel("PDF Watermark Removal Tool")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(); title_font.setPointSize(20); title_font.setBold(True)
        title_label.setFont(title_font); title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # --- Settings row ---
        settings_group = QGroupBox("Watermark Removal Settings")
        settings_layout = QHBoxLayout(settings_group)

        self.generic_checkbox = QCheckBox("Auto-detect and remove repeating text watermarks"); self.generic_checkbox.setChecked(True)
        self.vector_checkbox  = QCheckBox("Handle vector/object-based watermarks (path overlays)"); self.vector_checkbox.setChecked(True)
        self.nuke_preview_checkbox = QCheckBox("Nuke Preview (show XObjects-only output)"); self.nuke_preview_checkbox.setChecked(False)

        settings_layout.addWidget(self.generic_checkbox)
        settings_layout.addWidget(self.vector_checkbox)
        settings_layout.addWidget(self.nuke_preview_checkbox)
        settings_layout.addStretch()

        layout.addWidget(settings_group)

        splitter = QSplitter(Qt.Orientation.Horizontal); layout.addWidget(splitter)

        # Left panel
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)

        self.drop_zone = DropZoneWidget(); self.drop_zone.files_dropped.connect(self.handle_files_dropped)
        left_layout.addWidget(self.drop_zone)

        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing"); self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("""
            QPushButton { background-color: #007ACC; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #005fa3; }
            QPushButton:disabled { background-color: #cccccc; }
        """)

        # NEW: Non-XObject Nuke button
        self.nuke_button = QPushButton("Non-XObject Nuke"); self.nuke_button.setEnabled(False)
        self.nuke_button.clicked.connect(self.start_nuke_processing)
        self.nuke_button.setStyleSheet("""
            QPushButton { background-color: #6f42c1; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #59339d; }
            QPushButton:disabled { background-color: #cccccc; }
        """)

        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #cccccc; }
        """)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.nuke_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        progress_layout.addLayout(button_layout)
        left_layout.addWidget(progress_group)

        self.status_label = QLabel("Ready to process PDF files"); self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        left_layout.addWidget(self.status_label)

        self.log_text = QTextEdit(); self.log_text.setMaximumHeight(240); self.log_text.setPlaceholderText("Processing log will appear here...")
        left_layout.addWidget(self.log_text)

        splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        preview_label = QLabel("PDF Preview"); preview_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_layout.addWidget(preview_label)

        self.preview_widget = PDFPreviewWidget(); right_layout.addWidget(self.preview_widget)

        splitter.addWidget(right_panel); splitter.setSizes([560, 640])

        self.current_files = []

        # --- PREVIEW: clear first, then refresh on any setting change ---
        self.generic_checkbox.toggled.connect(self.on_any_setting_changed)
        self.vector_checkbox.toggled.connect(self.on_any_setting_changed)
        self.nuke_preview_checkbox.toggled.connect(self.on_any_setting_changed)

    def on_any_setting_changed(self, _=None):
        """Clear preview immediately, then regenerate (if exactly one file is selected)."""
        self.preview_widget.clear_preview("Refreshing preview…")
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.cancel()
        if len(self.current_files) == 1:
            self._start_preview(self.current_files[0])
        else:
            self.preview_widget.set_preview_disabled("Preview disabled when multiple PDFs are selected.")

    def handle_files_dropped(self, files):
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.cancel()

        self.current_files = files
        self.processed_files = []
        self.log_text.clear()

        file_count = len(files)
        self.drop_zone.setText(f"📁 {file_count} PDF file{'s' if file_count != 1 else ''} selected\n\nReady to process!")
        self.start_button.setEnabled(True)
        self.nuke_button.setEnabled(True)

        self.log_text.append(f"Selected {file_count} PDF file(s):")
        for f in files[:10]: self.log_text.append(f"  • {os.path.basename(f)}")
        if file_count > 10: self.log_text.append(f"  ... and {file_count - 10} more files")

        self.status_label.setText(f"Ready to process {file_count} PDF file(s)")

        if file_count == 1:
            first_file = files[0]
            self.preview_widget.update_preview(first_file, None)
            self._start_preview(first_file)
        else:
            self.preview_widget.set_preview_disabled("Preview disabled when multiple PDFs are selected.")

    def _start_preview(self, pdf_file):
        remove_generic = self.generic_checkbox.isChecked()
        handle_vector = self.vector_checkbox.isChecked()
        mode = 'nuke' if self.nuke_preview_checkbox.isChecked() else 'normal'

        self.preview_thread = PreviewGeneratorThread(
            pdf_file,
            remove_generic=remove_generic,
            handle_vector=handle_vector,
            cfg=self.cfg,
            mode=mode
        )
        self.preview_thread.status_updated.connect(self.status_label.setText)
        self.preview_thread.error_occurred.connect(self.on_preview_error)
        self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
        self.preview_thread.start()
        if mode == 'normal':
            self.log_text.append("🔍 Generating processed preview of the selected file…")
        else:
            self.log_text.append("🧨 Generating XObjects-only (Nuke) preview of the selected file…")

    def on_preview_error(self, file_path, error_message):
        filename = os.path.basename(file_path)
        self.log_text.append(f"❌ Preview error for {filename}: {error_message}")

    def start_processing(self):
        if not self.current_files:
            return
        self._start_processing_with_mode('normal')

    def start_nuke_processing(self):
        if not self.current_files:
            return
        self._start_processing_with_mode('nuke')

    def _start_processing_with_mode(self, mode='normal'):
        self.start_button.setEnabled(False)
        self.nuke_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        remove_generic = self.generic_checkbox.isChecked()
        handle_vector = self.vector_checkbox.isChecked()

        self.processor_thread = PDFProcessorThread(
            self.current_files,
            remove_generic=remove_generic,
            handle_vector=handle_vector,
            cfg=self.cfg,
            mode=mode
        )
        self.processor_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processor_thread.status_updated.connect(self.status_label.setText)
        self.processor_thread.file_processed.connect(self.on_file_processed)
        self.processor_thread.error_occurred.connect(self.on_error_occurred)
        self.processor_thread.processing_complete.connect(self.on_processing_complete)
        self.processor_thread.preview_ready.connect(self.preview_widget.update_preview)

        self.processor_thread.start()
        if mode == 'normal':
            self.log_text.append("\n--- Processing Started (Normal) ---")
        else:
            self.log_text.append("\n--- Processing Started (Non-XObject Nuke) ---")

    def cancel_processing(self):
        if self.processor_thread:
            self.processor_thread.cancel()
            self.cancel_button.setText("Cancelling…")
            self.cancel_button.setEnabled(False)

    def on_file_processed(self, original_path, output_path):
        filename = os.path.basename(original_path)
        self.log_text.append(f"✅ Successfully processed: {filename}")
        self.log_text.append(f"   Output: {os.path.basename(output_path)}")
        self.processed_files.append((original_path, output_path))

    def on_error_occurred(self, file_path, error_message):
        filename = os.path.basename(file_path)
        self.log_text.append(f"❌ Error processing {filename}: {error_message}")

    def on_processing_complete(self, processed_files):
        self.start_button.setEnabled(True)
        self.nuke_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        self.progress_bar.setValue(100)

        success_count = len(processed_files)
        total_count = len(self.current_files)

        self.log_text.append("\n--- Processing Complete ---")
        self.log_text.append(f"Successfully processed: {success_count}/{total_count} files")

        if processed_files:
            self.log_text.append("\nOutput files saved:")
            for original, output in processed_files:
                self.log_text.append(f"  • {os.path.basename(output)}")

        self.status_label.setText(f"Complete! Processed {success_count}/{total_count} files")

        if success_count > 0:
            QMessageBox.information(
                self, "Processing Complete",
                f"Successfully processed {success_count} out of {total_count} PDF files.\n\n"
                f"Normal mode uses '(No Watermarks)' and Nuke mode uses '(Only XObjects)' filename prefixes."
            )

        if success_count > 1:
            resp = QMessageBox.question(
                self,
                "Open All Processed PDFs",
                "Would you like to open all processed PDFs now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if resp == QMessageBox.StandardButton.Yes:
                for _, out_path in processed_files:
                    open_file_cross_platform(out_path)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PDF Watermark Remover")
    app.setApplicationVersion("1.2")
    app.setOrganizationName("PDF Tools")
    app.setStyle('Fusion')

    window = WatermarkRemovalApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()