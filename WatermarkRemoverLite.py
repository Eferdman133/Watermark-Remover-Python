#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Watermark Remover - Tkinter GUI (PyMuPDF-only)
- Non-destructive: edits watermark instructions / annots, preserves body text
- Options: confidential phrases, generic repeating/diagonal text, vector/object overlays
- Auto-preview (first page) for single file via temp processed copy
- Multi-file batch processing with progress, cancel, and optional open-all
"""

import os
import sys
import re
import math
import queue
import threading
import tempfile
import subprocess
import platform
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Checkbutton, BooleanVar, Text, Scrollbar, StringVar,
    filedialog, messagebox, BOTH, LEFT, RIGHT, X, Y, NSEW, W, E, N, S, TOP, BOTTOM
)
try:
    # Tk 8.6 supports PNGs via PhotoImage(file=...), which we use for preview.
    from tkinter import PhotoImage
except Exception:
    PhotoImage = None

import fitz  # PyMuPDF


# ------------------------------
# Cross-platform open helper
# ------------------------------
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


# ------------------------------
# Core removal engine (PyMuPDF-only)
# ------------------------------
class WatermarkRemover:
    @staticmethod
    def remove_watermarks(input_path: str,
                          output_path: str,
                          remove_confidential: bool = True,
                          remove_generic: bool = True,
                          handle_vector: bool = True,
                          status_cb=None) -> int:
        """
        Process one PDF and write a de-watermarked copy.
        Returns: number of removed watermark elements (heuristic count).
        """
        doc = fitz.open(input_path)
        if doc.is_encrypted:
            doc.close()
            raise RuntimeError("PDF is password-protected")

        removed = 0

        # 0) Vector/object overlays (outlined shapes/logos)
        if handle_vector:
            if status_cb: status_cb("Scanning vector/object overlays…")
            removed += WatermarkRemover._remove_vector_object_watermarks(doc)

        # 1) Specific phrases + safe annotation delete
        if remove_confidential:
            if status_cb: status_cb("Removing phrase-based & watermark annotations…")
            removed += WatermarkRemover._remove_confidential_watermarks(doc)

        # 2) Repeating text + diagonal/large text
        if remove_generic:
            if status_cb: status_cb("Detecting repeating/diagonal text watermarks…")
            removed += WatermarkRemover._remove_generic_watermarks(doc)

        # 3) Artifact-tagged watermark blocks
        if status_cb: status_cb("Removing artifact-tagged watermark blocks…")
        removed += WatermarkRemover._remove_artifact_watermarks(doc)

        # Save
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        return removed

    # ---------- VECTOR / OBJECT-BASED ----------
    @staticmethod
    def _remove_vector_object_watermarks(doc) -> int:
        """
        Heuristic: remove q..Q chunks that look like big path overlays:
        - Many path ops (m l c re h S s f f* B B* b b*)
        - Rotated transform (cm) ~20–70°
        - Often near end of stream (overlay); and/or page drawings cover a big area.
        Baked-in defaults (no knobs exposed).
        """
        stream_tail_fraction = 0.70
        min_vector_ops = 0
        min_vector_score = 0
        angle_min, angle_max = 20, 80
        area_frac_min = 0.0

        removed = 0

        # Precompute page drawings to estimate coverage
        drawings_meta = []
        for pno in range(len(doc)):
            try:
                page = doc.load_page(pno)
                drs = page.get_drawings()
            except Exception:
                drs = []
            union = None
            for d in drs:
                r = d.get("rect")
                if r:
                    union = r if union is None else union | r
            drawings_meta.append((drs, union))

        qQ_split = re.compile(r"(q.*?Q)", re.S)
        path_ops_re = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?)(?![A-Za-z])")
        text_ops_re = re.compile(r"BT|Tj|TJ")
        xobj_do_re = re.compile(r"/[A-Za-z0-9_.#-]+\s+Do")
        cm_re = re.compile(
            r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm"
        )

        def has_diag_cm(chunk: str) -> bool:
            for a, b, c, d, _, _ in cm_re.findall(chunk):
                try:
                    a, b = float(a), float(b)
                    ang = abs(math.degrees(math.atan2(b, a)))
                    if angle_min <= ang <= angle_max:
                        return True
                except Exception:
                    continue
            return False

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
            if len(parts) <= 1:
                continue

            page_area = max(1.0, page.rect.get_area())
            drs, union_rect = drawings_meta[pno]
            area_large_flag = False
            if union_rect:
                union_area_frac = union_rect.get_area() / page_area
                if union_area_frac >= area_frac_min and len(drs) >= 30:
                    area_large_flag = True

            changed = False
            out = []
            total_len = len(raw)
            running = 0
            i = 0
            while i < len(parts):
                chunk = parts[i]
                if i % 2 == 1:  # q..Q
                    vector_ops = len(path_ops_re.findall(chunk))
                    text_ops = len(text_ops_re.findall(chunk))
                    xobj_ops = len(xobj_do_re.findall(chunk))
                    score = vector_ops - text_ops - xobj_ops

                    pos_frac = (running + len(chunk) / 2.0) / float(total_len) if total_len > 0 else 0.0
                    near_tail = pos_frac >= stream_tail_fraction
                    diag = has_diag_cm(chunk)

                    if vector_ops >= min_vector_ops and ((diag and near_tail) or area_large_flag) and score >= min_vector_score:
                        # Drop the overlay chunk
                        removed += 1
                        changed = True
                        running += len(chunk)
                        i += 1
                        continue

                out.append(chunk)
                running += len(chunk)
                i += 1

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

    # ---------- SPECIFIC PHRASES + ANNOTS ----------
    @staticmethod
    def _remove_confidential_watermarks(doc) -> int:
        removed = 0
        phrases = [
            "CONFIDENTIAL INFORMATION - DO NOT DISTRIBUTE",
            "CONFIDENTIAL INFORMATION",
            "DO NOT DISTRIBUTE",
            "CONFIDENTIAL",
            "DRAFT",
            "INTERNAL USE ONLY",
        ]
        up = [p.upper() for p in phrases]

        # Annotations
        for page in doc:
            removed += WatermarkRemover._safe_delete_matching_annots(
                page, lambda a: WatermarkRemover._annot_matches_watermark(a, up)
            )

        # Text objects
        for page in doc:
            removed += WatermarkRemover._strip_text_objects(page, phrases)

        return removed

    # ---------- GENERIC REPEATING + DIAGONAL LARGE ----------
    @staticmethod
    def _remove_generic_watermarks(doc) -> int:
        if len(doc) < 2:
            return 0
        removed = 0
        cands = WatermarkRemover._detect_repeating_text(doc)

        for page in doc:
            removed += WatermarkRemover._safe_delete_matching_annots(
                page, lambda a: WatermarkRemover._annot_matches_watermark(a, None)
            )

        for page in doc:
            if cands:
                removed += WatermarkRemover._strip_text_objects(page, phrases=cands)
            removed += WatermarkRemover._strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60)

        return removed

    @staticmethod
    def _detect_repeating_text(doc):
        sample_pages = min(5, len(doc))
        text_positions = {}
        for i in range(sample_pages):
            page = doc.load_page(i)
            blocks = page.get_text("dict").get("blocks", [])
            for b in blocks:
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        t = (span.get("text") or "").strip()
                        if len(t) < 5:
                            continue
                        x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                        rx = (x0 + x1) / 2 / page.rect.width
                        ry = (y0 + y1) / 2 / page.rect.height
                        key = f"{rx:.2f},{ry:.2f}"
                        text_positions.setdefault(t, {})
                        text_positions[t][key] = text_positions[t].get(key, 0) + 1
        return [t for t, pos in text_positions.items() if max(pos.values()) >= min(3, int(sample_pages * 0.6))]

    # ---------- TEXT OBJECT STRIP ----------
    @staticmethod
    def _strip_text_objects(page, phrases) -> int:
        try:
            xrefs = page.get_contents()
            if not xrefs:
                return 0
            up = [p.upper() for p in phrases if p and p.strip()]
            if not up:
                return 0

            raw = page.read_contents().decode("latin-1", errors="ignore")

            # Also strip /Artifact /Watermark marked content inside BT/ET
            art_pat = re.compile(r"/Artifact\b[^B]*?/Watermark\b.*?BDC(.*?)EMC", re.S | re.I)
            raw2, n_art = art_pat.subn("", raw)

            parts = re.split(r"(BT.*?ET)", raw2, flags=re.S | re.I)
            changed = False
            out = []
            removed_blocks = 0

            tj_simple = re.compile(r"\((?:\\.|[^()])*\)\s*Tj", re.S)
            tj_array = re.compile(r"\[\s*(?:\((?:\\.|[^()])*\)\s*-?\d+\s*)+\]\s*TJ", re.S)

            def block_matches(block_text: str) -> bool:
                strings = []
                strings.extend(m.group(0) for m in tj_simple.finditer(block_text))
                strings.extend(m.group(0) for m in tj_array.finditer(block_text))
                if not strings:
                    return False

                def extract_literals(s):
                    lits = re.findall(r"\((?:\\.|[^()])*\)", s, flags=re.S)
                    out = []
                    for lit in lits:
                        t = lit[1:-1].replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
                        out.append(t)
                    return " ".join(out)

                merged = " ".join(extract_literals(s) for s in strings).upper()
                merged = re.sub(r"\s+", " ", merged)
                return any(p in merged for p in up)

            for i, chunk in enumerate(parts):
                if i % 2 == 1 and block_matches(chunk):
                    removed_blocks += 1
                    changed = True
                    continue
                out.append(chunk)

            if changed or n_art:
                new_stream = "".join(out).encode("latin-1", errors="ignore")
                if isinstance(xrefs, (list, tuple)) and xrefs:
                    page.parent.update_stream(xrefs[0], new_stream)
                    for xr in xrefs[1:]:
                        page.parent.update_stream(xr, b"")
                else:
                    page.parent.update_stream(xrefs, new_stream)
            return removed_blocks + n_art
        except Exception:
            return 0

    # ---------- DIAGONAL LARGE TEXT ----------
    @staticmethod
    def _strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60) -> int:
        removed = 0
        try:
            text = page.get_text("dict")
            if not text:
                return 0

            has_diag_large = False
            for block in text.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    d = line.get("dir")
                    if not d:
                        continue
                    ang = abs(math.degrees(math.atan2(d[1], d[0])))
                    if min_deg <= ang <= max_deg and any(span.get("size", 0) >= min_pt for span in line.get("spans", [])):
                        has_diag_large = True
                        break
                if has_diag_large:
                    break
            if not has_diag_large:
                return 0

            xrefs = page.get_contents()
            if not xrefs:
                return 0
            raw = page.read_contents().decode("latin-1", errors="ignore")
            parts = re.split(r"(BT.*?ET)", raw, flags=re.S | re.I)
            changed = False
            out = []

            for i, chunk in enumerate(parts):
                if i % 2 == 1:
                    if "Tj" in chunk or "TJ" in chunk:
                        removed += 1
                        changed = True
                        continue
                out.append(chunk)

            if changed:
                new_stream = "".join(out).encode("latin-1", errors="ignore")
                if isinstance(xrefs, (list, tuple)) and xrefs:
                    page.parent.update_stream(xrefs[0], new_stream)
                    for xr in xrefs[1:]:
                        page.parent.update_stream(xr, b"")
                else:
                    page.parent.update_stream(xrefs, new_stream)
            return removed
        except Exception:
            return 0

    # ---------- ARTIFACT-TAGGED ----------
    @staticmethod
    def _remove_artifact_watermarks(doc) -> int:
        removed = 0
        for page in doc:
            page.clean_contents()
            xref = page.get_contents()[0] if page.get_contents() else None
            if not xref:
                continue
            try:
                lines = page.read_contents().splitlines()
                out = []
                skipping = False
                for ln in lines:
                    s = ln.decode("latin-1", errors="ignore")
                    if "/Artifact" in s and "/Watermark" in s:
                        skipping = True
                        removed += 1
                        continue
                    if skipping and "EMC" in s:
                        skipping = False
                        continue
                    if not skipping:
                        out.append(ln)
                if len(out) != len(lines):
                    page.parent.update_stream(xref, b"\n".join(out))
            except Exception:
                pass
        return removed

    # ---------- ANNOTATIONS ----------
    @staticmethod
    def _safe_delete_matching_annots(page, predicate) -> int:
        removed = 0
        try:
            annot = page.first_annot
        except Exception:
            annot = None

        while annot:
            try:
                nxt = annot.next
            except Exception:
                nxt = None
            try:
                if predicate(annot):
                    try:
                        page.delete_annot(annot)
                        removed += 1
                    except Exception:
                        pass
            except Exception:
                pass
            annot = nxt
        return removed

    @staticmethod
    def _annot_matches_watermark(annot, phrases_upper=None) -> bool:
        try:
            t = (annot.type[1] or "").upper()
        except Exception:
            t = ""
        info = getattr(annot, "info", {}) or {}
        contents = str(info.get("content", "")).upper()
        name = str(info.get("name", "")).upper()
        if t in {"STAMP", "WATERMARK"}:
            return True
        if phrases_upper:
            for p in phrases_upper:
                if p in contents or p in name:
                    return True
        return False


# ------------------------------
# Background worker
# ------------------------------
class ProcessorThread(threading.Thread):
    def __init__(self, files, opts, ui_callback, done_callback, cancel_flag):
        super().__init__(daemon=True)
        self.files = files
        self.opts = opts
        self.ui_callback = ui_callback
        self.done_callback = done_callback
        self.cancel_flag = cancel_flag
        self.processed = []

    def run(self):
        total = len(self.files)
        for i, src in enumerate(self.files, 1):
            if self.cancel_flag.is_set():
                break
            name = os.path.basename(src)
            out = str(Path(src).parent / f"(No Watermarks) {Path(src).name}")

            def status(msg):
                self.ui_callback(("status", f"{msg}"))

            try:
                self.ui_callback(("status", f"Processing {name}…"))
                n = WatermarkRemover.remove_watermarks(
                    src, out,
                    remove_confidential=self.opts["confidential"],
                    remove_generic=self.opts["generic"],
                    handle_vector=self.opts["vector"],
                    status_cb=status
                )
                self.processed.append((src, out))
                self.ui_callback(("log", f"✅ {name} → removed ~{n} watermark elements\n    ↳ {os.path.basename(out)}"))
            except Exception as e:
                self.ui_callback(("log", f"❌ {name}: {e}"))

            pct = int(i / total * 100)
            self.ui_callback(("progress", pct))

            # For the first processed file, request preview update
            if i == 1 and not self.cancel_flag.is_set():
                self.ui_callback(("preview-final", (src, out)))

        self.done_callback(self.processed)


# ------------------------------
# Tkinter GUI
# ------------------------------
class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Watermark Remover Lite by Emil Ferdman (With Help from ChatGPT")
        self.geometry("1100x720")

        # State
        self.files = []
        self.preview_imgs = {"orig": None, "proc": None}
        self.preview_paths = {"orig": None, "proc": None}
        self.worker = None
        self.cancel_flag = threading.Event()
        self.ui_queue = queue.Queue()

        # Options
        self.var_confidential = BooleanVar(value=True)
        self.var_generic = BooleanVar(value=True)
        self.var_vector = BooleanVar(value=True)

        self._build_ui()
        self._poll_ui_queue()

    # ---------- UI Layout ----------
    def _build_ui(self):
        # Top bar
        top = Frame(self); top.pack(side=TOP, fill=X, padx=10, pady=10)

        Label(top, text="Watermark Removal Settings", font=("Segoe UI", 12, "bold")).pack(side=LEFT, padx=(0, 20))
        Checkbutton(top, text="Remove 'CONFIDENTIAL' phrases", variable=self.var_confidential).pack(side=LEFT, padx=5)
        Checkbutton(top, text="Auto-detect repeating/diagonal", variable=self.var_generic).pack(side=LEFT, padx=5)
        Checkbutton(top, text="Handle vector/object overlays", variable=self.var_vector).pack(side=LEFT, padx=5)

        # Middle: left (controls/log) and right (preview)
        mid = Frame(self); mid.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Left panel
        left = Frame(mid); left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))

        # File controls
        ctrl = Frame(left); ctrl.pack(fill=X)
        Button(ctrl, text="Select PDFs…", command=self.on_select_files).pack(side=LEFT)
        Button(ctrl, text="Start", command=self.on_start, state="disabled", name="startbtn").pack(side=LEFT, padx=6)
        Button(ctrl, text="Cancel", command=self.on_cancel, state="disabled", name="cancelbtn").pack(side=LEFT)
        self.progress_var = StringVar(value="0%")
        self.progress_lbl = Label(ctrl, textvariable=self.progress_var); self.progress_lbl.pack(side=RIGHT)

        # Status
        self.status_var = StringVar(value="Ready.")
        Label(left, textvariable=self.status_var, fg="#2e7d32").pack(fill=X, pady=(6, 4))

        # Log area
        log_frame = Frame(left); log_frame.pack(fill=BOTH, expand=True)
        self.log_text = Text(log_frame, height=18, wrap="word")
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        sb = Scrollbar(log_frame, command=self.log_text.yview)
        sb.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=sb.set)

        # Right panel: Preview
        right = Frame(mid, bd=1, relief="sunken"); right.pack(side=RIGHT, fill=BOTH, expand=True)

        Label(right, text="Preview (page 1)", font=("Segoe UI", 12, "bold")).pack(anchor=W, padx=8, pady=(8, 0))
        pv = Frame(right); pv.pack(fill=BOTH, expand=True, padx=8, pady=8)

        # Original
        self.orig_label = Label(pv, text="Original will appear here", bd=1, relief="groove", width=45, height=28, anchor="center")
        self.orig_label.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))
        self.orig_label.bind("<Button-1>", lambda e: self._open_preview_pdf("orig"))

        # Processed
        self.proc_label = Label(pv, text="Processed will appear here", bd=1, relief="groove", width=45, height=28, anchor="center")
        self.proc_label.pack(side=RIGHT, fill=BOTH, expand=True, padx=(6, 0))
        self.proc_label.bind("<Button-1>", lambda e: self._open_preview_pdf("proc"))

    # ---------- Events ----------
    def on_select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select PDF files", filetypes=[("PDF files", "*.pdf")]
        )
        if not paths:
            return
        self.files = list(paths)
        self._log(f"Selected {len(self.files)} PDF(s):")
        for p in self.files[:10]:
            self._log("  • " + os.path.basename(p))
        if len(self.files) > 10:
            self._log(f"  ... and {len(self.files) - 10} more")

        self._set_status(f"Ready to process {len(self.files)} file(s)")
        self._set_btn_state("startbtn", True)
        self._set_btn_state("cancelbtn", False)
        self._set_progress(0)

        # Preview policy
        if len(self.files) == 1:
            self._make_preview(self.files[0])
        else:
            self._clear_preview("Preview disabled when multiple PDFs are selected.")

    def on_start(self):
        if not self.files:
            return
        self.cancel_flag.clear()
        self._set_btn_state("startbtn", False)
        self._set_btn_state("cancelbtn", True)
        self._set_progress(0)

        opts = dict(
            confidential=self.var_confidential.get(),
            generic=self.var_generic.get(),
            vector=self.var_vector.get()
        )
        self._log("\n--- Processing Started ---")

        def ui_callback(msg):
            # msg is a tuple like ("status", text) or ("progress", pct) ...
            self.ui_queue.put(msg)

        def done_callback(processed):
            # Called in worker thread at the end
            self.ui_queue.put(("done", processed))

        self.worker = ProcessorThread(self.files, opts, ui_callback, done_callback, self.cancel_flag)
        self.worker.start()

    def on_cancel(self):
        if self.worker and self.worker.is_alive():
            self.cancel_flag.set()
            self._set_btn_state("cancelbtn", False)
            self._set_status("Cancelling…")

    # ---------- Preview helpers ----------
    def _make_preview(self, pdf_path):
        """Generate preview by processing to a temp PDF and rendering page 1 PNGs via PyMuPDF."""
        self._clear_preview("Generating preview…")
        try:
            base = os.path.basename(pdf_path)
            tmp_proc = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")

            # Run with current options
            opts = dict(
                remove_confidential=self.var_confidential.get(),
                remove_generic=self.var_generic.get(),
                handle_vector=self.var_vector.get(),
            )
            WatermarkRemover.remove_watermarks(pdf_path, tmp_proc, **opts, status_cb=lambda msg: None)

            # Render original & processed page 1 to temp PNGs
            orig_png = self._render_page_to_png(pdf_path)
            proc_png = self._render_page_to_png(tmp_proc)

            self.preview_paths["orig"] = pdf_path
            self.preview_paths["proc"] = tmp_proc

            if PhotoImage:
                if orig_png and os.path.exists(orig_png):
                    self.preview_imgs["orig"] = PhotoImage(file=orig_png)
                    self.orig_label.config(image=self.preview_imgs["orig"], text="")
                else:
                    self.orig_label.config(image="", text="Original preview unavailable")

                if proc_png and os.path.exists(proc_png):
                    self.preview_imgs["proc"] = PhotoImage(file=proc_png)
                    self.proc_label.config(image=self.preview_imgs["proc"], text="")
                else:
                    self.proc_label.config(image="", text="Processed preview unavailable")
            else:
                self.orig_label.config(text="Preview requires Tk PhotoImage support")
                self.proc_label.config(text="Preview requires Tk PhotoImage support")
        except Exception as e:
            self._clear_preview(f"Preview error: {e}")

    def _render_page_to_png(self, pdf_path, page_index=0, scale=1.5):
        """Use PyMuPDF to render a page to a temp PNG file."""
        try:
            doc = fitz.open(pdf_path)
            if page_index >= len(doc):
                page_index = 0
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            out = os.path.join(tempfile.gettempdir(), f"__preview_{os.path.basename(pdf_path)}_{page_index}.png")
            pix.save(out)
            doc.close()
            return out
        except Exception:
            return None

    def _open_preview_pdf(self, which):
        path = self.preview_paths.get(which)
        if path:
            open_file_cross_platform(path)

    def _clear_preview(self, msg="Preview will appear here"):
        self.preview_imgs["orig"] = None
        self.preview_imgs["proc"] = None
        self.preview_paths["orig"] = None
        self.preview_paths["proc"] = None
        self.orig_label.config(image="", text=msg)
        self.proc_label.config(image="", text=msg)

    # ---------- UI helpers ----------
    def _set_btn_state(self, name, enable: bool):
        try:
            btn = self.nametowidget(name)
            btn.config(state=("normal" if enable else "disabled"))
        except Exception:
            pass

    def _set_status(self, text):
        self.status_var.set(text)

    def _set_progress(self, pct: int):
        self.progress_var.set(f"{pct}%")

    def _log(self, text: str):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    # ---------- UI queue polling (thread-safe updates) ----------
    def _poll_ui_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind, payload = msg[0], msg[1] if len(msg) > 1 else None
                if kind == "status":
                    self._set_status(payload)
                elif kind == "progress":
                    self._set_progress(int(payload))
                elif kind == "log":
                    self._log(payload)
                elif kind == "preview-final":
                    # Update right-side processed with first finished output
                    src, out = payload
                    self.preview_paths["orig"] = src
                    self.preview_paths["proc"] = out
                    # Refresh processed image only (optional)
                    proc_png = self._render_page_to_png(out)
                    if proc_png and PhotoImage:
                        self.preview_imgs["proc"] = PhotoImage(file=proc_png)
                        self.proc_label.config(image=self.preview_imgs["proc"], text="")
                elif kind == "done":
                    processed = payload or []
                    self._on_done(processed)
        except queue.Empty:
            pass
        self.after(80, self._poll_ui_queue)

    def _on_done(self, processed):
        self._set_btn_state("startbtn", True)
        self._set_btn_state("cancelbtn", False)
        self._set_progress(100)

        succ = len(processed)
        total = len(self.files)
        self._log("\n--- Processing Complete ---")
        self._log(f"Successfully processed: {succ}/{total} files")
        if processed:
            self._log("\nOutput files saved:")
            for _, out in processed:
                self._log(f"  • {os.path.basename(out)}")
        self._set_status(f"Complete! Processed {succ}/{total} files")

        if succ > 0:
            messagebox.showinfo(
                "Processing Complete",
                f"Successfully processed {succ} out of {total} PDF files.\n\n"
                f"Processed files are saved with '(No Watermarks) ' prefix next to the originals."
            )

        if succ > 1:
            if messagebox.askyesno("Open All Processed PDFs", "Would you like to open all processed PDFs now?"):
                for _, out_path in processed:
                    open_file_cross_platform(out_path)


# ------------------------------
# Main
# ------------------------------
def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
