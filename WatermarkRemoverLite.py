#!/usr/bin/env python3
# PDF Watermark Remover — Lite GUI (Tkinter + PyMuPDF only)
# - Single dependency: PyMuPDF (fitz)
# - Features: repeating text removal, vector/object watermarks (XObject hashing + q..Q),
#             live preview (first page), progress bar, log, open outputs.

import os
import sys
import math
import re
import platform
import subprocess
import tempfile
import hashlib
import threading
import queue
import base64
from pathlib import Path

import fitz  # PyMuPDF

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "PDF Watermark Remover (Lite)"
PREVIEW_W, PREVIEW_H = 400, 500


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


class WatermarkRemover:
    """Static watermark removal helpers."""

    @staticmethod
    def remove_watermarks(
        input_path, output_path,
        remove_generic=True,
        handle_vector=True,
        status_cb=None
    ):
        doc = fitz.open(input_path)
        if doc.is_encrypted:
            doc.close()
            raise RuntimeError("PDF is password-protected")

        total_removed = 0

        if handle_vector:
            if status_cb: status_cb("Scanning for repeated vector/object watermarks…")
            total_removed += WatermarkRemover._remove_vector_object_watermarks(doc)

        if remove_generic:
            if status_cb: status_cb("Detecting & removing generic repeating text watermarks…")
            total_removed += WatermarkRemover._remove_generic_watermarks(doc)

        if status_cb: status_cb("Cleaning artifact-based watermark blocks…")
        total_removed += WatermarkRemover._remove_artifact_watermarks(doc)

        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        return total_removed

    # ---------- VECTOR / OBJECT-BASED WATERMARK (XObject hashing + q..Q majority) ----------
    @staticmethod
    def _remove_vector_object_watermarks(
        doc,
        sample_pages=5,
        min_vector_ops_sample=5,
        min_vector_ops_remove=5,
        repeat_threshold=0.8,        # appear on ≥ 80% of sampled pages
    ):
        if len(doc) == 0:
            return 0

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

        # Stage A: repeated XObjects (by stream hash)
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

            name_to_xref = {}
            try:
                xobjs = page.get_xobjects()
                # handle dict or tuple returns across PyMuPDF versions
                for it in xobjs:
                    if isinstance(it, dict):
                        nm = it.get("name")
                        xr = it.get("xref")
                        if nm is None or xr is None:
                            continue
                        if isinstance(nm, bytes):
                            nm = nm.decode("latin-1", "ignore")
                        nm = str(nm).lstrip("/")
                        name_to_xref[nm] = xr
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
                if text_ops_re.search(s):
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

        for h in repeated_hashes:
            xr = xref_by_hash.get(h)
            if not xr:
                continue
            try:
                doc.update_stream(xr, b"")  # blank the form XObject
                removed += 1
            except Exception:
                pass

        # Stage B: repeated q..Q blocks (vector-heavy, no text), majority threshold
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
                if i % 2 == 1:
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
                            # ensure it's not tiny noise
                            vops = len(path_ops_re.findall(chunk))
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
    def _remove_generic_watermarks(doc):
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
            removed += WatermarkRemover._strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60)
        return removed

    @staticmethod
    def _detect_repeating_text_patterns(doc):
        sample_pages = min(5, len(doc))
        text_positions = {}
        for page_num in range(sample_pages):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = (span.get("text") or "").strip()
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
                strings += [m.group(0) for m in tj_simple.finditer(block_text)]
                strings += [m.group(0) for m in tj_array.finditer(block_text)]
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
    def _strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60):
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


# --------------------------- GUI (Tkinter) ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x750")
        try:
            self.iconbitmap(default='')  # no-op if missing
        except Exception:
            pass

        # State
        self.current_files = []
        self.processed_files = []
        self.preview_photo_orig = None
        self.preview_photo_proc = None
        self.preview_tmpfile = None
        self.worker = None
        self.queue = queue.Queue()

        # UI
        self._build_ui()
        self._poll_queue()

    # ---- UI Layout ----
    def _build_ui(self):
        # Title
        title = ttk.Label(self, text="PDF Watermark Removal Tool (Lite)", anchor="center")
        title.configure(font=("Segoe UI", 18, "bold"))
        title.pack(pady=10)

        # Settings
        frame_settings = ttk.LabelFrame(self, text="Watermark Removal Settings")
        frame_settings.pack(fill="x", padx=20, pady=10)

        self.var_generic = tk.BooleanVar(value=True)
        self.var_vector = tk.BooleanVar(value=True)

        chk_generic = ttk.Checkbutton(frame_settings, text="Auto-detect and remove repeating text watermarks", variable=self.var_generic, command=self._on_settings_changed)
        chk_vector  = ttk.Checkbutton(frame_settings, text="Handle vector/object-based watermarks (path overlays)", variable=self.var_vector, command=self._on_settings_changed)
        chk_generic.pack(side="left", padx=10, pady=8)
        chk_vector.pack(side="left", padx=10, pady=8)

        # Main Split
        frame_main = ttk.Frame(self)
        frame_main.pack(fill="both", expand=True, padx=20, pady=10)
        frame_left = ttk.Frame(frame_main)
        frame_right = ttk.Frame(frame_main)
        frame_left.pack(side="left", fill="both", expand=True)
        frame_right.pack(side="left", fill="both", expand=True)

        # Left controls
        lf_select = ttk.LabelFrame(frame_left, text="Select PDFs")
        lf_select.pack(fill="x", pady=8)
        self.lbl_drop = ttk.Label(lf_select, text="No files selected.")
        self.lbl_drop.pack(side="left", padx=10, pady=8)
        btn_browse = ttk.Button(lf_select, text="Browse…", command=self._select_files)
        btn_browse.pack(side="right", padx=10, pady=8)

        lf_progress = ttk.LabelFrame(frame_left, text="Processing")
        lf_progress.pack(fill="x", pady=8)
        self.progress = ttk.Progressbar(lf_progress, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=10, pady=10)
        frm_btns = ttk.Frame(lf_progress); frm_btns.pack(fill="x", padx=10, pady=4)
        self.btn_start = ttk.Button(frm_btns, text="Start Processing", command=self._start_processing, state="disabled")
        self.btn_cancel = ttk.Button(frm_btns, text="Cancel", command=self._cancel_processing, state="disabled")
        self.btn_start.pack(side="left", padx=4)
        self.btn_cancel.pack(side="left", padx=4)

        self.lbl_status = ttk.Label(frame_left, text="Ready.")
        self.lbl_status.pack(fill="x", pady=6, padx=2)

        lf_log = ttk.LabelFrame(frame_left, text="Log")
        lf_log.pack(fill="both", expand=True)
        self.txt_log = tk.Text(lf_log, height=12, wrap="word")
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)

        # Right previews
        lf_preview = ttk.LabelFrame(frame_right, text="Preview (first page)")
        lf_preview.pack(fill="both", expand=True, padx=0, pady=0)

        top = ttk.Frame(lf_preview); top.pack(fill="x")
        ttk.Label(top, text="Original").pack(side="left", padx=10, pady=6)
        ttk.Label(top, text="Processed").pack(side="right", padx=10, pady=6)

        body = ttk.Frame(lf_preview); body.pack(fill="both", expand=True)
        self.canvas_orig = tk.Label(body, anchor="center", relief="groove")
        self.canvas_proc = tk.Label(body, anchor="center", relief="groove")
        self.canvas_orig.place(relx=0.02, rely=0.08, width=PREVIEW_W, height=PREVIEW_H)
        self.canvas_proc.place(relx=0.55, rely=0.08, width=PREVIEW_W, height=PREVIEW_H)

        bottom = ttk.Frame(lf_preview); bottom.pack(fill="x")
        self.btn_open_orig = ttk.Button(bottom, text="Open Original", command=self._open_original, state="disabled")
        self.btn_open_proc = ttk.Button(bottom, text="Open Processed", command=self._open_processed, state="disabled")
        self.btn_open_orig.pack(side="left", padx=10, pady=6)
        self.btn_open_proc.pack(side="right", padx=10, pady=6)

    # ---- Event handlers ----
    def _select_files(self):
        files = filedialog.askopenfilenames(title="Select PDF Files", filetypes=[("PDF Files","*.pdf")])
        if not files:
            return
        self.current_files = list(files)
        self.processed_files = []
        self._log(f"Selected {len(files)} file(s).")
        self.lbl_drop.config(text=f"{len(files)} file(s) selected.")
        self.btn_start.config(state="normal")
        self.lbl_status.config(text=f"Ready to process {len(files)} file(s).")

        if len(files) == 1:
            # show original preview and start processed preview
            self._render_preview(self.current_files[0], which="orig")
            self._refresh_preview()
        else:
            self._clear_preview("Preview disabled when multiple PDFs are selected.")

    def _on_settings_changed(self):
        # Requirement: clear preview first, then refresh
        self._clear_preview("Refreshing preview…")
        self._refresh_preview()

    def _start_processing(self):
        if not self.current_files:
            return
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress["value"] = 0
        self._log("\n--- Processing Started ---")
        self._set_status("Processing…")

        t = threading.Thread(target=self._worker_process, daemon=True)
        self.worker = t
        t.start()

    def _cancel_processing(self):
        # Best-effort: we don't kill threads; we just mark intent and stop queuing further tasks
        self._set_status("Cancel requested (will stop after current file).")
        self.worker = None  # signal to worker loop

    def _open_original(self):
        if self.current_files:
            open_file_cross_platform(self.current_files[0])

    def _open_processed(self):
        if self.processed_files:
            open_file_cross_platform(self.processed_files[0][1])

    # ---- Worker threads ----
    def _worker_process(self):
        files = self.current_files[:]
        total = len(files)
        processed = []
        for i, pdf in enumerate(files):
            if self.worker is None:
                break
            try:
                in_path = Path(pdf)
                out_path = in_path.parent / f"(No Watermarks) {in_path.name}"
                def _status(msg):
                    self.queue.put(("status", msg))
                WatermarkRemover.remove_watermarks(
                    str(in_path), str(out_path),
                    remove_generic=self.var_generic.get(),
                    handle_vector=self.var_vector.get(),
                    status_cb=_status
                )
                processed.append((str(in_path), str(out_path)))
                self.queue.put(("file_done", (str(in_path), str(out_path))))
                if i == 0 and len(files) >= 1:
                    self.queue.put(("preview_ready", (str(in_path), str(out_path))))
            except Exception as e:
                self.queue.put(("error", (pdf, f"Processing error: {e}")))
            self.queue.put(("progress", int(((i + 1) / total) * 100)))

        self.queue.put(("complete", processed))

    def _worker_preview(self, in_file, out_file):
        try:
            def _status(msg):
                self.queue.put(("status", f"[Preview] {msg}"))
            WatermarkRemover.remove_watermarks(
                in_file, out_file,
                remove_generic=self.var_generic.get(),
                handle_vector=self.var_vector.get(),
                status_cb=_status
            )
            self.queue.put(("preview_done", (in_file, out_file)))
        except Exception as e:
            self.queue.put(("preview_error", (in_file, f"Preview error: {e}")))

    # ---- Queue / UI updates ----
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "file_done":
                    in_path, out_path = payload
                    self.processed_files.append((in_path, out_path))
                    self._log(f"✅ Processed: {os.path.basename(in_path)}")
                    self._log(f"   Output: {os.path.basename(out_path)}")
                elif kind == "preview_ready":
                    in_path, out_path = payload
                    # update processed side preview
                    self._render_preview(in_path, which="orig")
                    self._render_preview(out_path, which="proc")
                elif kind == "complete":
                    processed = payload
                    self._on_complete(processed)
                elif kind == "error":
                    f, err = payload
                    self._log(f"❌ Error processing {os.path.basename(f)}: {err}")
                elif kind == "preview_done":
                    in_path, out_path = payload
                    self._render_preview(in_path, which="orig")
                    self._render_preview(out_path, which="proc")
                elif kind == "preview_error":
                    f, err = payload
                    self._log(f"❌ {err}")
                else:
                    pass
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)

    def _on_complete(self, processed_files):
        self.btn_start.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.progress["value"] = 100
        success = len(processed_files)
        total = len(self.current_files)
        self._log("\n--- Processing Complete ---")
        self._log(f"Successfully processed: {success}/{total} files")
        if processed_files:
            self._log("\nOutput files saved:")
            for _, outp in processed_files:
                self._log(f"  • {os.path.basename(outp)}")
        self._set_status(f"Complete! Processed {success}/{total} files")
        if success > 0:
            messagebox.showinfo("Processing Complete",
                                f"Successfully processed {success} out of {total} PDF files.\n\n"
                                f"Processed files are saved with '(No Watermarks)' prefix in the same directory.")
        if success > 1:
            if messagebox.askyesno("Open All Processed PDFs", "Open all processed PDFs now?"):
                for _, outp in processed_files:
                    open_file_cross_platform(outp)

    # ---- Preview helpers ----
    def _clear_preview(self, msg=""):
        # Clear first (requirement)
        self.canvas_orig.config(image="", text=msg)
        self.canvas_proc.config(image="", text=msg)
        self.preview_photo_orig = None
        self.preview_photo_proc = None
        self.btn_open_orig.config(state="disabled")
        self.btn_open_proc.config(state="disabled")

    def _refresh_preview(self):
        if len(self.current_files) != 1:
            return
        in_file = self.current_files[0]
        self.preview_tmpfile = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {os.path.basename(in_file)}")
        # kick background preview
        t = threading.Thread(target=self._worker_preview, args=(in_file, self.preview_tmpfile), daemon=True)
        t.start()

    def _pix_to_photoimage(self, pix: fitz.Pixmap):
        # Use PNG bytes -> base64 -> PhotoImage(data=...)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return tk.PhotoImage(data=b64)

    def _render_preview(self, pdf_path, which="orig", page_num=0):
        if not pdf_path or not os.path.exists(pdf_path):
            return
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc): page_num = 0
            page = doc.load_page(page_num)
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            photo = self._pix_to_photoimage(pix)
            doc.close()
        except Exception:
            return

        if which == "orig":
            self.preview_photo_orig = photo
            self.canvas_orig.config(image=self.preview_photo_orig, text="")
            self.btn_open_orig.config(state="normal")
        else:
            self.preview_photo_proc = photo
            self.canvas_proc.config(image=self.preview_photo_proc, text="")
            self.btn_open_proc.config(state="normal")

    # ---- Misc helpers ----
    def _set_status(self, text):
        self.lbl_status.config(text=text)

    def _log(self, text):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")


def main():
    # Basic HiDPI fix for Windows
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore
        except Exception:
            pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
