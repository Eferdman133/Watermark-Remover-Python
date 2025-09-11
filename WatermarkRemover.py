# Professional PDF Watermark Removal Application
# (Non-destructive + Safe Annot Deletion + Smart Preview + Open-All-When-Done)
# New: Optional handling for vector/object-based watermarks (path-heavy overlays).

import sys
import os
import math
import re
import subprocess
import platform
from pathlib import Path
import tempfile

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


class WatermarkRemover:
    """Static helper for watermark removal—used by both preview and processing threads."""

    @staticmethod
    def remove_watermarks(input_path, output_path,
                          remove_confidential=True,
                          remove_generic=True,
                          handle_vector=False,
                          status_cb=None):
        try:
            doc = fitz.open(input_path)
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            total_removed = 0

            # 0) Optional: vector/object watermark handler (path-heavy overlays)
            if handle_vector:
                if status_cb: status_cb("Scanning for vector/object-based watermarks…")
                total_removed += WatermarkRemover._remove_vector_object_watermarks(doc)

            # 1) Specific phrases + safe annot deletion
            if remove_confidential:
                if status_cb: status_cb("Removing specific 'CONFIDENTIAL' watermarks…")
                total_removed += WatermarkRemover._remove_confidential_watermarks(doc)

            # 2) Generic repeating + diagonal big text
            if remove_generic:
                if status_cb: status_cb("Detecting & removing generic repeating watermarks…")
                total_removed += WatermarkRemover._remove_generic_watermarks(doc)

            # 3) Artifact-marked content
            if status_cb: status_cb("Cleaning artifact-based watermark blocks…")
            total_removed += WatermarkRemover._remove_artifact_watermarks(doc)

            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            return total_removed
        except Exception:
            raise

    # ---------- VECTOR / OBJECT-BASED WATERMARK PASS ----------
    @staticmethod
    def _remove_vector_object_watermarks(doc,
                                         stream_tail_fraction=0.70,
                                         min_vector_ops=0,
                                         min_vector_score=0,
                                         angle_min=20, angle_max=80,
                                         area_frac_min=0.0):
        """
        Heuristic removal of vector (path) overlays typically used for outline watermarks:
        - Split each page stream into q..Q chunks (graphics state groups).
        - Score each chunk:
            * path_ops = count of path operators (m,l,c,re,h,S,s,f,f*,B,B*,b,b*)
            * has_diagonal = a 'cm' rotation ≈ 20–70 degrees (or -20..-70)
            * near_tail    = chunk appears in last 30% of the page stream
            * area_large   = union bbox of paths covers >= area_frac_min of the page (via page.get_drawings)
              (approximate: if many drawings exist and union bbox is large)
        - If (path_ops >= min_vector_ops) AND ((has_diagonal and near_tail) OR area_large)
          => remove that q..Q chunk from the content stream.
        """
        removed = 0

        # Precompute drawing info (vector outlines) to estimate large-area overlays
        drawings_cache = []
        for pno in range(len(doc)):
            try:
                page = doc.load_page(pno)
                drs = page.get_drawings()  # list of path objects with 'rect', 'items', 'fill', 'stroke'...
            except Exception:
                drs = []
            union = None
            for d in drs:
                r = d.get("rect")
                if not r:
                    # try to build from points
                    pts = []
                    for it in d.get("items", []):
                        for seg in it:
                            if isinstance(seg, fitz.Point):
                                pts.append(seg)
                    if pts:
                        xs = [p.x for p in pts]
                        ys = [p.y for p in pts]
                        r = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                if r:
                    union = r if union is None else union | r
            drawings_cache.append((drs, union))

        # Regex helpers
        qQ_split = re.compile(r"(q.*?Q)", re.S)
        path_ops_re = re.compile(r"(?<![A-Za-z])(?:m|l|c|re|h|S|s|f\*?|B\*?|b\*?)(?![A-Za-z])")
        text_ops_re = re.compile(r"BT|Tj|TJ")
        xobj_do_re = re.compile(r"/[A-Za-z0-9_.#-]+\s+Do")
        cm_re = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+cm")

        def _has_diag_cm(chunk: str) -> bool:
            for a,b,c,d,_,_ in cm_re.findall(chunk):
                try:
                    a,b,c,d = float(a), float(b), float(c), float(d)
                    # angle of x-axis
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
            drs, union_rect = drawings_cache[pno]
            area_large_flag = False
            if union_rect:
                union_area_frac = union_rect.get_area() / page_area
                if union_area_frac >= area_frac_min and len(drs) >= 30:
                    # many drawing items and large area: typical outline watermark
                    area_large_flag = True

            changed = False
            out = []
            total_len = len(raw)

            # We’ll track offset while re-assembling to estimate tail-ness based on index
            # Compute cumulative lengths to find the original position of each chunk.
            # Simpler heuristic: use running length from beginning while rebuilding.
            running = 0
            i = 0
            while i < len(parts):
                chunk = parts[i]
                if i % 2 == 1:
                    # this is a q..Q block (graphics state chunk)
                    vector_ops = len(path_ops_re.findall(chunk))
                    text_ops   = len(text_ops_re.findall(chunk))
                    xobj_ops   = len(xobj_do_re.findall(chunk))
                    score = vector_ops - text_ops - xobj_ops

                    # position of this chunk as fraction in original stream (approx with running+len(chunk)/total)
                    pos_frac = (running + len(chunk)/2.0) / float(total_len) if total_len > 0 else 0.0
                    near_tail = pos_frac >= stream_tail_fraction

                    has_diag = _has_diag_cm(chunk)

                    if vector_ops >= min_vector_ops and ( (has_diag and near_tail) or area_large_flag ) and score >= min_vector_score:
                        # Drop this vector overlay block
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

    # ---------- Original passes ----------
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

    @staticmethod
    def _remove_generic_watermarks(doc):
        if len(doc) < 2: return 0
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


class PDFProcessorThread(QThread):
    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(list)
    preview_ready = pyqtSignal(str, str)

    def __init__(self, pdf_files,
                 remove_confidential=True,
                 remove_generic=True,
                 handle_vector=False):
        super().__init__()
        self.pdf_files = pdf_files
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cancel_requested = False
        self.processed_files = []

    def run(self):
        total_files = len(self.pdf_files)
        for i, pdf_file in enumerate(self.pdf_files):
            if self.cancel_requested:
                break
            try:
                self.status_updated.emit(f"Processing {os.path.basename(pdf_file)}...")
                input_path = Path(pdf_file)
                output_path = input_path.parent / f"(No Watermarks) {input_path.name}"

                def _status(msg): self.status_updated.emit(msg)

                WatermarkRemover.remove_watermarks(
                    str(input_path), str(output_path),
                    remove_confidential=self.remove_confidential,
                    remove_generic=self.remove_generic,
                    handle_vector=self.handle_vector,
                    status_cb=_status
                )

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
                 remove_confidential=True,
                 remove_generic=True,
                 handle_vector=False):
        super().__init__()
        self.pdf_file = pdf_file
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
        self.handle_vector = handle_vector
        self.cancel_requested = False
        self.tmp_output = None

    def run(self):
        try:
            base = os.path.basename(self.pdf_file)
            self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")

            def _status(msg): self.status_updated.emit(f"[Preview] {msg}")

            WatermarkRemover.remove_watermarks(
                self.pdf_file, self.tmp_output,
                remove_confidential=self.remove_confidential,
                remove_generic=self.remove_generic,
                handle_vector=self.handle_vector,
                status_cb=_status
            )
            if not self.cancel_requested:
                self.preview_ready.emit(self.pdf_file, self.tmp_output)
        except Exception as e:
            if not self.cancel_requested:
                self.error_occurred.emit(self.pdf_file, f"Preview error: {e}")

    def cancel(self):
        self.cancel_requested = True


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


class WatermarkRemovalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor_thread = None
        self.preview_thread = None
        self.processed_files = []

        self.init_ui()
        self.setWindowTitle("PDF Watermark Remover by Emil Ferdman (With Help from ChatGPT)")
        self.setGeometry(100, 100, 1200, 800)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass

    def init_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget); layout.setSpacing(15); layout.setContentsMargins(20,20,20,20)

        title_label = QLabel("PDF Watermark Removal Tool")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(); title_font.setPointSize(20); title_font.setBold(True)
        title_label.setFont(title_font); title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        settings_group = QGroupBox("Watermark Removal Settings")
        settings_layout = QHBoxLayout(settings_group)

        self.confidential_checkbox = QCheckBox("Remove 'CONFIDENTIAL' watermarks"); self.confidential_checkbox.setChecked(True)
        self.generic_checkbox = QCheckBox("Auto-detect and remove repeating watermarks"); self.generic_checkbox.setChecked(True)
        self.vector_checkbox = QCheckBox("Handle vector/object-based watermarks (path overlays)"); self.vector_checkbox.setChecked(True)

        settings_layout.addWidget(self.confidential_checkbox)
        settings_layout.addWidget(self.generic_checkbox)
        settings_layout.addWidget(self.vector_checkbox)

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
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #cccccc; }
        """)

        button_layout.addWidget(self.start_button); button_layout.addWidget(self.cancel_button); button_layout.addStretch()
        progress_layout.addLayout(button_layout)
        left_layout.addWidget(progress_group)

        self.status_label = QLabel("Ready to process PDF files"); self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        left_layout.addWidget(self.status_label)

        self.log_text = QTextEdit(); self.log_text.setMaximumHeight(220); self.log_text.setPlaceholderText("Processing log will appear here...")
        left_layout.addWidget(self.log_text)

        splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        preview_label = QLabel("PDF Preview"); preview_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_layout.addWidget(preview_label)

        self.preview_widget = PDFPreviewWidget(); right_layout.addWidget(self.preview_widget)

        splitter.addWidget(right_panel); splitter.setSizes([600, 600])

        self.current_files = []

    def handle_files_dropped(self, files):
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.cancel()

        self.current_files = files
        self.processed_files = []
        self.log_text.clear()

        file_count = len(files)
        self.drop_zone.setText(f"📁 {file_count} PDF file{'s' if file_count != 1 else ''} selected\n\nReady to process!")
        self.start_button.setEnabled(True)

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
        remove_confidential = self.confidential_checkbox.isChecked()
        remove_generic = self.generic_checkbox.isChecked()
        handle_vector = self.vector_checkbox.isChecked()

        self.preview_thread = PreviewGeneratorThread(
            pdf_file,
            remove_confidential=remove_confidential,
            remove_generic=remove_generic,
            handle_vector=handle_vector,
        )
        self.preview_thread.status_updated.connect(self.status_label.setText)
        self.preview_thread.error_occurred.connect(self.on_preview_error)
        self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
        self.preview_thread.start()
        self.log_text.append("🔍 Generating processed preview of the selected file…")

    def on_preview_error(self, file_path, error_message):
        filename = os.path.basename(file_path)
        self.log_text.append(f"❌ Preview error for {filename}: {error_message}")

    def start_processing(self):
        if not self.current_files:
            return
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        remove_confidential = self.confidential_checkbox.isChecked()
        remove_generic = self.generic_checkbox.isChecked()
        handle_vector = self.vector_checkbox.isChecked()

        self.processor_thread = PDFProcessorThread(
            self.current_files,
            remove_confidential=remove_confidential,
            remove_generic=remove_generic,
            handle_vector=handle_vector,
        )
        self.processor_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processor_thread.status_updated.connect(self.status_label.setText)
        self.processor_thread.file_processed.connect(self.on_file_processed)
        self.processor_thread.error_occurred.connect(self.on_error_occurred)
        self.processor_thread.processing_complete.connect(self.on_processing_complete)
        self.processor_thread.preview_ready.connect(self.preview_widget.update_preview)

        self.processor_thread.start()
        self.log_text.append("\n--- Processing Started ---")

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
                f"Processed files are saved with '(No Watermarks)' prefix in the same directory."
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
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PDF Tools")
    app.setStyle('Fusion')

    window = WatermarkRemovalApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()