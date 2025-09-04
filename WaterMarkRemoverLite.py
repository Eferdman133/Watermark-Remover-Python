#!/usr/bin/env python3
# watermark_gui_tk.py
# GUI watermark remover using tkinter + PyMuPDF only (no PyQt6, no Pillow)

import os
import sys
import math
import re
import base64
import tempfile
import threading
import queue
import platform
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import fitz  # PyMuPDF


# ------------------------------------------------------------
# Cross-platform "open file" helper
# ------------------------------------------------------------
def open_file_cross_platform(path: str):
    try:
        if not path or not os.path.exists(path):
            return
        system = platform.system()
        if system == "Darwin":      # macOS
            subprocess.call(["open", path])
        elif system == "Windows":   # Windows
            os.startfile(path)  # type: ignore[attr-defined]
        else:                       # Linux and others
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


# ============================================================
# Core non-destructive watermark remover (no white-out)
# ============================================================
class WatermarkRemover:
    """Standalone, non-destructive remover (annotations + content stream surgery)."""

    @staticmethod
    def remove_watermarks(input_path, output_path, remove_confidential=True, remove_generic=True, status_cb=None):
        try:
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            if remove_confidential:
                if status_cb: status_cb("Removing targeted 'CONFIDENTIAL' watermarks…")
                total_removed += WatermarkRemover._remove_confidential_watermarks(doc)

            if remove_generic:
                if status_cb: status_cb("Detecting & removing generic repeating/diagonal watermarks…")
                total_removed += WatermarkRemover._remove_generic_watermarks(doc)

            if status_cb: status_cb("Cleaning artifact / marked-content watermark blocks…")
            total_removed += WatermarkRemover._remove_artifact_watermarks(doc)

            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return total_removed

        except Exception:
            raise

    # ---------- Targeted removal ----------
    @staticmethod
    def _remove_confidential_watermarks(doc):
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
        for page in doc:
            removed += WatermarkRemover._safe_delete_matching_annots(
                page, lambda a: WatermarkRemover._annot_matches_watermark(a, up)
            )
        for page in doc:
            removed += WatermarkRemover._strip_text_objects(page, phrases)
        return removed

    # ---------- Generic repeating / diagonal ----------
    @staticmethod
    def _remove_generic_watermarks(doc):
        if len(doc) < 2:
            return 0
        removed = 0
        candidates = WatermarkRemover._detect_repeating_text_patterns(doc)
        for page in doc:
            removed += WatermarkRemover._safe_delete_matching_annots(
                page, lambda a: WatermarkRemover._annot_matches_watermark(a, None)
            )
        for page in doc:
            if candidates:
                removed += WatermarkRemover._strip_text_objects(page, candidates)
            removed += WatermarkRemover._strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60)
        return removed

    @staticmethod
    def _detect_repeating_text_patterns(doc):
        sample_pages = min(5, len(doc))
        text_positions = {}
        for i in range(sample_pages):
            page = doc.load_page(i)
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        if len(text) < 5 or text.lower() in {"the","and","or","of","to","in","for"}:
                            continue
                        bbox = span["bbox"]
                        rel_x = (bbox[0] + bbox[2]) / 2 / page.rect.width
                        rel_y = (bbox[1] + bbox[3]) / 2 / page.rect.height
                        key = f"{rel_x:.2f},{rel_y:.2f}"
                        text_positions.setdefault(text, {})
                        text_positions[text][key] = text_positions[text].get(key, 0) + 1
        candidates = []
        for text, pos in text_positions.items():
            if max(pos.values()) >= min(3, int(sample_pages * 0.6)):
                candidates.append(text)
        return candidates

    # ---------- Content stream surgery (BT…ET text) ----------
    @staticmethod
    def _strip_text_objects(page, phrases):
        try:
            if not page.get_contents():
                return 0
            up_phrases = [p.upper() for p in phrases if p and p.strip()]
            if not up_phrases:
                return 0

            raw = page.read_contents().decode("latin-1", errors="ignore")

            # Remove marked-content watermark blocks
            artifact_pattern = re.compile(r"/Artifact\b[^B]*?/Watermark\b.*?BDC(.*?)EMC", re.S | re.I)
            raw2, n_art = artifact_pattern.subn("", raw)

            parts = re.split(r"(BT.*?ET)", raw2, flags=re.S | re.I)
            changed = False
            out = []
            removed_blocks = 0

            tj_simple = re.compile(r"\((?:\\.|[^()])*\)\s*Tj", re.S)
            tj_array  = re.compile(r"\[\s*(?:\((?:\\.|[^()])*\)\s*-?\d+\s*)+\]\s*TJ", re.S)

            def block_has_phrase(chunk: str) -> bool:
                strings = list(tj_simple.finditer(chunk)) + list(tj_array.finditer(chunk))
                if not strings:
                    return False
                def extract_literals(s: str) -> str:
                    lits = re.findall(r"\((?:\\.|[^()])*\)", s, flags=re.S)
                    out_l = []
                    for lit in lits:
                        t = lit[1:-1]
                        t = t.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
                        out_l.append(t)
                    return " ".join(out_l)
                merged = " ".join(extract_literals(m.group(0)) for m in strings).upper()
                merged = re.sub(r"\s+", " ", merged)
                return any(p in merged for p in up_phrases)

            for i, chunk in enumerate(parts):
                if i % 2 == 1:  # BT...ET block
                    if block_has_phrase(chunk):
                        removed_blocks += 1
                        changed = True
                        continue
                out.append(chunk)

            if changed or n_art:
                new_stream = "".join(out).encode("latin-1", errors="ignore")
                xrefs = page.get_contents()
                if isinstance(xrefs, (list, tuple)):
                    if xrefs:
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
            info = page.get_text("dict")
            blocks = info.get("blocks", []) if info else []
            has_candidates = False
            for block in blocks:
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

            raw = page.read_contents().decode("latin-1", errors="ignore")
            parts = re.split(r"(BT.*?ET)", raw, flags=re.S | re.I)
            changed = False
            out = []

            tm_rot = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm", re.I)
            tf_big = re.compile(r"/[A-Za-z0-9]+\s+(\d{2,})\s+Tf", re.I)

            for i, chunk in enumerate(parts):
                if i % 2 == 1:  # BT...ET
                    has_big = any(float(s) >= min_pt for s in tf_big.findall(chunk))
                    rot_hit = False
                    for a,b,c,d,_,_ in tm_rot.findall(chunk):
                        try:
                            a,b,c,d = float(a), float(b), float(c), float(d)
                            ang = abs(math.degrees(math.atan2(b, a)))
                            if min_deg <= ang <= max_deg:
                                rot_hit = True
                                break
                        except:
                            pass
                    if has_big and rot_hit:
                        removed += 1
                        changed = True
                        continue
                out.append(chunk)

            if changed:
                new_stream = "".join(out).encode("latin-1", errors="ignore")
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

    # ----------------------------
    # SAFE ANNOTATION HELPERS
    # ----------------------------
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
                        line_str = line.decode("latin-1", errors="ignore")
                        if "/Artifact" in line_str and "/Watermark" in line_str:
                            skip_until_emc = True
                            removed_count += 1
                            continue
                        if skip_until_emc and "EMC" in line_str:
                            skip_until_emc = False
                            continue
                        if not skip_until_emc:
                            modified_lines.append(line)
                    if len(modified_lines) != len(content_lines):
                        page.parent.update_stream(xref, b"\n".join(modified_lines))
                except Exception:
                    pass
        return removed_count


# ============================================================
# Worker threads (no Pillow / no PyQt): use threading + queue
# ============================================================
class ProcessorThread(threading.Thread):
    def __init__(self, files, remove_confidential, remove_generic, ui_queue, cancel_flag):
        super().__init__(daemon=True)
        self.files = files
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
        self.ui_queue = ui_queue
        self.cancel_flag = cancel_flag
        self.processed = []

    def run(self):
        total = len(self.files)
        for idx, f in enumerate(self.files):
            if self.cancel_flag.is_set():
                break
            try:
                self.ui_queue.put(("status", f"Processing {os.path.basename(f)}…"))
                inp = Path(f)
                outp = inp.parent / f"(No Watermarks) {inp.name}"

                def _status(msg):
                    self.ui_queue.put(("status", msg))

                WatermarkRemover.remove_watermarks(
                    str(inp),
                    str(outp),
                    remove_confidential=self.remove_confidential,
                    remove_generic=self.remove_generic,
                    status_cb=_status
                )
                self.processed.append((str(inp), str(outp)))
                self.ui_queue.put(("file_done", (str(inp), str(outp))))
            except Exception as e:
                self.ui_queue.put(("error", (f, f"Processing error: {e}")))
            self.ui_queue.put(("progress", int(((idx + 1) / total) * 100)))
        self.ui_queue.put(("complete", self.processed))


class PreviewThread(threading.Thread):
    """Builds a processed preview to temp for a single file."""
    def __init__(self, file, remove_confidential, remove_generic, ui_queue, cancel_flag):
        super().__init__(daemon=True)
        self.file = file
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
        self.ui_queue = ui_queue
        self.cancel_flag = cancel_flag
        self.tmp_output = None

    def run(self):
        try:
            if self.cancel_flag.is_set():
                return
            base = os.path.basename(self.file)
            self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")

            def _status(msg):
                self.ui_queue.put(("status", f"[Preview] {msg}"))

            WatermarkRemover.remove_watermarks(
                self.file,
                self.tmp_output,
                remove_confidential=self.remove_confidential,
                remove_generic=self.remove_generic,
                status_cb=_status
            )
            if not self.cancel_flag.is_set():
                self.ui_queue.put(("preview_ready", (self.file, self.tmp_output)))
        except Exception as e:
            if not self.cancel_flag.is_set():
                self.ui_queue.put(("error", (self.file, f"Preview error: {e}")))


# ============================================================
# Tkinter GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Watermark Remover (tkinter)")
        self.geometry("1100x720")

        self.files = []
        self.processed_files = []
        self.processor_thread = None
        self.preview_thread = None
        self.ui_queue = queue.Queue()
        self.cancel_flag = threading.Event()

        self._build_ui()
        self._poll_queue()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Title
        title = ttk.Label(self, text="PDF Watermark Removal Tool", font=("Segoe UI", 16, "bold"))
        title.pack(pady=(10, 5))

        # Settings
        settings = ttk.Frame(self)
        settings.pack(fill="x", padx=16)

        self.var_conf = tk.BooleanVar(value=True)
        self.var_gen = tk.BooleanVar(value=True)

        ttk.Checkbutton(settings, text="Remove 'CONFIDENTIAL' watermarks", variable=self.var_conf).pack(side="left", padx=6)
        ttk.Checkbutton(settings, text="Auto-detect & remove repeating watermarks", variable=self.var_gen).pack(side="left", padx=6)

        # Main split
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=8)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(8,0))

        # Left: file controls
        ctl = ttk.Frame(left)
        ctl.pack(fill="x")
        ttk.Button(ctl, text="Add PDF Files…", command=self.add_files).pack(side="left")
        ttk.Button(ctl, text="Clear List", command=self.clear_files).pack(side="left", padx=6)

        self.files_list = tk.Listbox(left, height=10, activestyle="dotbox")
        self.files_list.pack(fill="x", pady=6)

        # Progress & buttons
        prog = ttk.Frame(left)
        prog.pack(fill="x", pady=(6, 0))

        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=(0, 80))
        self.btn_start = ttk.Button(prog, text="Start Processing", command=self.start_processing, state="disabled")
        self.btn_start.pack(side="right")
        self.btn_cancel = ttk.Button(prog, text="Cancel", command=self.cancel_processing, state="disabled")
        self.btn_cancel.pack(side="right", padx=6)

        self.status = ttk.Label(left, text="Ready to process PDF files", foreground="#2e7d32")
        self.status.pack(anchor="w", pady=(6, 2))

        self.log = tk.Text(left, height=12, wrap="word")
        self.log.pack(fill="both", expand=True)

        # Right: Preview (Original / Processed Preview)
        prev_header = ttk.Label(right, text="Preview", font=("Segoe UI", 12, "bold"))
        prev_header.pack(anchor="w")

        self.preview_notice = ttk.Label(right, text="", foreground="#444")
        self.preview_notice.pack(anchor="w")

        pv = ttk.Frame(right)
        pv.pack(fill="both", expand=True)

        self.original_label = ttk.Label(pv, text="Original preview will appear here", anchor="center")
        self.processed_label = ttk.Label(pv, text="Processed (Preview) will appear here", anchor="center")

        # Use two stacked frames with labels
        self.original_label.pack(fill="both", expand=True, pady=(4, 4))
        self.processed_label.pack(fill="both", expand=True, pady=(4, 4))

        # Click handlers to open PDFs
        self.original_label.bind("<Button-1>", lambda e: self._open_current_pdf(which="orig"))
        self.processed_label.bind("<Button-1>", lambda e: self._open_current_pdf(which="proc"))

        # Keep PhotoImage refs
        self._orig_photo = None
        self._proc_photo = None
        self._current_preview_paths = (None, None)

        # Footer
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(self, text="Tip: Click a preview to open the full PDF.", foreground="#555").pack(pady=(0, 8))

    # ------------- File actions -------------
    def add_files(self):
        files = filedialog.askopenfilenames(
            title="Select PDF Files", filetypes=[("PDF Files", "*.pdf")])
        if not files:
            return
        self.files = list(files)
        self.files_list.delete(0, "end")
        for f in self.files:
            self.files_list.insert("end", f)
        self.btn_start.config(state="normal")
        self._log(f"Selected {len(self.files)} file(s).")
        self._update_preview_logic()

    def clear_files(self):
        self.files = []
        self.files_list.delete(0, "end")
        self.btn_start.config(state="disabled")
        self._set_preview_disabled("No file selected.")
        self._log("Cleared list.")

    # ------------- Preview handling -------------
    def _update_preview_logic(self):
        # Cancel any ongoing preview
        self._cancel_preview_thread()

        if len(self.files) == 0:
            self._set_preview_disabled("No file selected.")
            return

        if len(self.files) > 1:
            self._set_preview_disabled("Preview disabled when multiple PDFs are selected.")
            return

        # Exactly one file → show original immediately, kick off processed preview
        first = self.files[0]
        self._show_original_preview(first)
        self._show_processed_placeholder()

        # Start background processed preview
        self.preview_cancel_flag = threading.Event()
        self.preview_thread = PreviewThread(
            first,
            remove_confidential=self.var_conf.get(),
            remove_generic=self.var_gen.get(),
            ui_queue=self.ui_queue,
            cancel_flag=self.preview_cancel_flag
        )
        self.preview_thread.start()
        self._log("Generating processed preview of the selected file…")

    def _set_preview_disabled(self, msg: str):
        self.preview_notice.config(text=msg)
        self._set_preview_images(None, None)
        self._current_preview_paths = (None, None)

    def _show_original_preview(self, path: str):
        self.preview_notice.config(text="")
        self._orig_photo = self._render_page_to_photo(path)
        if self._orig_photo:
            self.original_label.config(image=self._orig_photo, text="")
        else:
            self.original_label.config(image="", text="Original preview will appear here")
        self._current_preview_paths = (path, self._current_preview_paths[1])

    def _show_processed_placeholder(self):
        self._proc_photo = None
        self.processed_label.config(image="", text="Processed (Preview) will appear here")

    def _set_processed_preview(self, preview_path: str):
        self._proc_photo = self._render_page_to_photo(preview_path)
        if self._proc_photo:
            self.processed_label.config(image=self._proc_photo, text="")
        else:
            self.processed_label.config(image="", text="Processed (Preview) will appear here")
        self._current_preview_paths = (self._current_preview_paths[0], preview_path)

    def _set_preview_images(self, orig_photo, proc_photo):
        self._orig_photo = orig_photo
        self._proc_photo = proc_photo
        if orig_photo is None:
            self.original_label.config(image="", text="Original preview will appear here")
        if proc_photo is None:
            self.processed_label.config(image="", text="Processed (Preview) will appear here")

    def _render_page_to_photo(self, pdf_path, page_num=0, max_w=520, scale_base=1.5):
        try:
            if not pdf_path or not os.path.exists(pdf_path):
                return None
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                page_num = 0
            page = doc.load_page(page_num)

            # compute scale to fit width
            # start with 1.5x, then adjust if still wider than max_w
            mat = fitz.Matrix(scale_base, scale_base)
            pix = page.get_pixmap(matrix=mat)
            if pix.width > max_w:
                factor = max_w / pix.width
                mat = fitz.Matrix(scale_base * factor, scale_base * factor)
                pix = page.get_pixmap(matrix=mat)

            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            # Tk PhotoImage supports PNG via Tk 8.6+; pass base64 data directly
            photo = tk.PhotoImage(data=b64)
            doc.close()
            return photo
        except Exception:
            try:
                doc.close()
            except Exception:
                pass
            return None

    def _open_current_pdf(self, which: str):
        orig, proc = self._current_preview_paths
        if which == "orig" and orig:
            open_file_cross_platform(orig)
        elif which == "proc" and proc:
            open_file_cross_platform(proc)

    def _cancel_preview_thread(self):
        if hasattr(self, "preview_cancel_flag") and self.preview_cancel_flag:
            self.preview_cancel_flag.set()

    # ------------- Processing -------------
    def start_processing(self):
        if not self.files:
            return
        self.cancel_flag.clear()
        self.processed_files = []
        self.progress["value"] = 0
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self._log("\n--- Processing Started ---")

        self.processor_thread = ProcessorThread(
            self.files,
            remove_confidential=self.var_conf.get(),
            remove_generic=self.var_gen.get(),
            ui_queue=self.ui_queue,
            cancel_flag=self.cancel_flag
        )
        self.processor_thread.start()

    def cancel_processing(self):
        self.cancel_flag.set()
        self.btn_cancel.config(state="disabled")
        self._log("Cancelling…")

    # ------------- UI Queue Polling -------------
    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.ui_queue.get_nowait()
                if msg == "status":
                    self.status.config(text=str(payload))
                elif msg == "progress":
                    self.progress["value"] = int(payload)
                elif msg == "file_done":
                    orig, outp = payload
                    self.processed_files.append((orig, outp))
                    self._log(f"✅ Processed: {os.path.basename(orig)} -> {os.path.basename(outp)}")
                elif msg == "error":
                    fpath, emsg = payload
                    self._log(f"❌ Error: {os.path.basename(fpath)} — {emsg}")
                elif msg == "complete":
                    self._on_complete(payload)
                elif msg == "preview_ready":
                    orig, preview_path = payload
                    # Only show if still one file selected and it's the same original
                    if len(self.files) == 1 and self.files[0] == orig:
                        self._set_processed_preview(preview_path)
                # mark as task done if needed (not essential for now)
        except queue.Empty:
            pass
        # schedule next poll
        self.after(100, self._poll_queue)

    def _on_complete(self, processed_list):
        self.btn_start.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.progress["value"] = 100

        success_count = len(processed_list)
        total = len(self.files)
        self._log("\n--- Processing Complete ---")
        self._log(f"Successfully processed: {success_count}/{total} files")

        if processed_list:
            self._log("\nOutput files saved:")
            for _, outp in processed_list:
                self._log(f"  • {os.path.basename(outp)}")

        self.status.config(text=f"Complete! Processed {success_count}/{total} files")

        # Standard info
        if success_count > 0:
            messagebox.showinfo(
                "Processing Complete",
                f"Successfully processed {success_count} out of {total} PDF files.\n\n"
                f"Processed files are saved with '(No Watermarks)' prefix in the same directory."
            )

        # If more than one, ask to open all
        if success_count > 1:
            if messagebox.askyesno("Open All Processed PDFs", "Would you like to open all processed PDFs now?"):
                for _, outp in processed_list:
                    open_file_cross_platform(outp)

    # ------------- Logging -------------
    def _log(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
