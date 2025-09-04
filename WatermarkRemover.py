# Professional PDF Watermark Removal Application
# (Non-destructive + Safe Annot Deletion + Smart Preview + Open-All-When-Done)

import sys
import os
import math
import re
import subprocess
import platform
from pathlib import Path
import tempfile
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QSplitter,
    QScrollArea, QMessageBox, QFileDialog,
    QGroupBox, QCheckBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal
)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QPixmap,
    QFont, QIcon
)

# PDF processing imports
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
# Core removal logic extracted into a reusable helper class
# ============================================================
class WatermarkRemover:
    """Static helper for watermark removal—used by both preview and processing threads."""

    @staticmethod
    def remove_watermarks(input_path, output_path, remove_confidential=True, remove_generic=True, status_cb=None):
        """Advanced watermark removal using PyMuPDF (non-destructive to underlying content)"""
        try:
            doc = fitz.open(input_path)

            # Check if PDF is encrypted
            if doc.is_encrypted:
                raise RuntimeError("PDF is password-protected")

            watermarks_found = 0

            # Step 1: Remove specific "CONFIDENTIAL" watermarks (no redactions)
            if remove_confidential:
                if status_cb: status_cb("Removing specific 'CONFIDENTIAL' watermarks…")
                watermarks_found += WatermarkRemover._remove_confidential_watermarks(doc)

            # Step 2: Detect and remove generic repeating watermarks (no redactions)
            if remove_generic:
                if status_cb: status_cb("Detecting and removing generic repeating watermarks…")
                watermarks_found += WatermarkRemover._remove_generic_watermarks(doc)

            # Step 3: Remove artifact-based watermarks
            if status_cb: status_cb("Cleaning artifact-based watermark blocks…")
            watermarks_found += WatermarkRemover._remove_artifact_watermarks(doc)

            # Save the processed document
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()

            return watermarks_found

        except Exception:
            raise

    # ----------------------------
    # WATERMARK REMOVAL (UPDATED)
    # ----------------------------

    @staticmethod
    def _remove_confidential_watermarks(doc):
        """Non-destructive removal of specific watermark phrases by editing text objects and deleting watermark/stamp annotations."""
        removed_count = 0

        confidential_phrases = [
            "CONFIDENTIAL INFORMATION - DO NOT DISTRIBUTE",
            "CONFIDENTIAL INFORMATION",
            "DO NOT DISTRIBUTE",
            "CONFIDENTIAL",
            "DRAFT",
            "INTERNAL USE ONLY",
        ]

        # Pass 1: delete stamp/watermark annotations outright (these are overlays) — SAFE walker
        phrases_upper = [p.upper() for p in confidential_phrases]
        for page in doc:
            removed_count += WatermarkRemover._safe_delete_matching_annots(
                page,
                lambda a: WatermarkRemover._annot_matches_watermark(a, phrases_upper)
            )

        # Pass 2: strip matching text objects from the page content stream (keeps underlying text intact)
        for page in doc:
            removed_count += WatermarkRemover._strip_text_objects(page, phrases=confidential_phrases)

        return removed_count

    @staticmethod
    def _remove_generic_watermarks(doc):
        """Detect repeating text candidates and remove by editing only the watermark text objects (non-destructive)."""
        if len(doc) < 2:
            return 0

        removed_count = 0
        watermark_candidates = WatermarkRemover._detect_repeating_text_patterns(doc)

        # Delete stamp/watermark annotations that don't have explicit phrases — SAFE walker
        for page in doc:
            removed_count += WatermarkRemover._safe_delete_matching_annots(
                page,
                lambda a: WatermarkRemover._annot_matches_watermark(a, None)
            )

        # Remove candidate strings from content streams (no white-out)
        for page in doc:
            if watermark_candidates:
                removed_count += WatermarkRemover._strip_text_objects(page, phrases=watermark_candidates)

            # Heuristic: remove big, diagonal text blocks often used as watermarks
            removed_count += WatermarkRemover._strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60)

        return removed_count

    @staticmethod
    def _detect_repeating_text_patterns(doc):
        """Detect text that appears on multiple pages in similar positions"""
        sample_pages = min(5, len(doc))
        text_positions = {}

        for page_num in range(sample_pages):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()

                            if len(text) < 5 or text.lower() in ["the", "and", "or", "of", "to", "in", "for"]:
                                continue

                            bbox = span["bbox"]
                            rel_x = (bbox[0] + bbox[2]) / 2 / page.rect.width
                            rel_y = (bbox[1] + bbox[3]) / 2 / page.rect.height

                            position_key = f"{rel_x:.2f},{rel_y:.2f}"

                            if text not in text_positions:
                                text_positions[text] = {}

                            text_positions[text][position_key] = text_positions[text].get(position_key, 0) + 1

        watermark_candidates = []
        for text, positions in text_positions.items():
            max_occurrences = max(positions.values())
            if max_occurrences >= min(3, sample_pages * 0.6):
                watermark_candidates.append(text)

        return watermark_candidates

    @staticmethod
    def _strip_text_objects(page, phrases):
        """
        Remove BT ... ET text objects that display any of the given phrases.
        Non-destructive to underlying page content (we're not painting a rectangle).
        """
        try:
            if not page.get_contents():
                return 0

            up_phrases = [p.upper() for p in phrases if p and p.strip()]
            if not up_phrases:
                return 0

            raw = page.read_contents().decode('latin-1', errors='ignore')

            # Remove marked-content watermark blocks
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
    def _strip_diagonal_large_text(page, min_pt=24, min_deg=30, max_deg=60):
        """Remove diagonal, large text BT...ET blocks (typical watermarks) without redactions."""
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
                                rot_hit = True
                                break
                        except:
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

    # ----------------------------
    # SAFE ANNOTATION HELPERS
    # ----------------------------
    @staticmethod
    def _safe_delete_matching_annots(page, predicate):
        """
        Walk annotations safely: always fetch `next_annot` BEFORE attempting deletion.
        `predicate(annot)` should return True if the annot must be deleted.
        """
        removed = 0
        try:
            annot = page.first_annot
        except Exception:
            annot = None

        while annot:
            # Always get next before deleting
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
        """Predicate used by safe annotation deletion."""
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
        """Remove PDF artifact-based watermarks (/Artifact /Watermark ... BDC ... EMC)"""
        removed_count = 0

        for page in doc:
            # Clean the page content streams
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


# ============================================================
# Threads: processing and preview
# ============================================================
class PDFProcessorThread(QThread):
    """Worker thread for batch processing to prevent GUI freezing"""

    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)
    processing_complete = pyqtSignal(list)
    preview_ready = pyqtSignal(str, str)

    def __init__(self, pdf_files, remove_confidential=True, remove_generic=True):
        super().__init__()
        self.pdf_files = pdf_files
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
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

                def _status(msg):
                    self.status_updated.emit(msg)

                WatermarkRemover.remove_watermarks(
                    str(input_path),
                    str(output_path),
                    remove_confidential=self.remove_confidential,
                    remove_generic=self.remove_generic,
                    status_cb=_status
                )

                self.file_processed.emit(pdf_file, str(output_path))
                self.processed_files.append((pdf_file, str(output_path)))

                if i == 0:
                    # Update preview with the first truly processed file
                    self.preview_ready.emit(pdf_file, str(output_path))

                progress = int(((i + 1) / total_files) * 100)
                self.progress_updated.emit(progress)

            except Exception as e:
                self.error_occurred.emit(pdf_file, f"Processing error: {str(e)}")

        self.processing_complete.emit(self.processed_files)

    def cancel(self):
        self.cancel_requested = True


class PreviewGeneratorThread(QThread):
    """Generates a non-destructive processed preview of a single file to a temp path."""

    preview_ready = pyqtSignal(str, str)  # original, processed_preview_path
    error_occurred = pyqtSignal(str, str)  # file_path, error
    status_updated = pyqtSignal(str)

    def __init__(self, pdf_file, remove_confidential=True, remove_generic=True):
        super().__init__()
        self.pdf_file = pdf_file
        self.remove_confidential = remove_confidential
        self.remove_generic = remove_generic
        self.cancel_requested = False
        self.tmp_output = None

    def run(self):
        try:
            base = os.path.basename(self.pdf_file)
            self.tmp_output = os.path.join(tempfile.gettempdir(), f"(Preview - No Watermarks) {base}")

            def _status(msg):
                self.status_updated.emit(f"[Preview] {msg}")

            WatermarkRemover.remove_watermarks(
                self.pdf_file,
                self.tmp_output,
                remove_confidential=self.remove_confidential,
                remove_generic=self.remove_generic,
                status_cb=_status
            )
            if not self.cancel_requested:
                self.preview_ready.emit(self.pdf_file, self.tmp_output)
        except Exception as e:
            if not self.cancel_requested:
                self.error_occurred.emit(self.pdf_file, f"Preview error: {e}")

    def cancel(self):
        self.cancel_requested = True


# ============================================================
# UI Widgets
# ============================================================
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PDFPreviewWidget(QScrollArea):
    """Widget for previewing PDF pages with before/after comparison"""

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

        self.current_pdfs = (None, None)  # (original_path, processed_path)

        self.original_label.clicked.connect(lambda: open_file_cross_platform(self.current_pdfs[0]))
        self.processed_label.clicked.connect(lambda: open_file_cross_platform(self.current_pdfs[1]))

    def set_preview_disabled(self, reason_text: str):
        """Show a disabled/notice state in the preview pane."""
        self.current_pdfs = (None, None)
        self.original_label.setPixmap(QPixmap())
        self.processed_label.setPixmap(QPixmap())
        self.original_label.setText(reason_text)
        self.processed_label.setText(reason_text)

    def update_preview(self, original_path, processed_path):
        """Update the preview with before/after pages"""
        self.current_pdfs = (original_path, processed_path)

        try:
            original_pixmap = self._render_pdf_page(original_path, 0)
            processed_pixmap = self._render_pdf_page(processed_path, 0)

            if original_pixmap:
                self.original_label.setPixmap(original_pixmap.scaled(
                    400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                self.original_label.setPixmap(QPixmap())
                self.original_label.setText("Original PDF will appear here")

            if processed_pixmap:
                self.processed_label.setPixmap(processed_pixmap.scaled(
                    400, 500, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                self.processed_label.setPixmap(QPixmap())
                self.processed_label.setText("Processed PDF will appear here")

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
            img_data = pix.tobytes("png")
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)
            doc.close()
            return pixmap
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


# ============================================================
# Main Window
# ============================================================
class WatermarkRemovalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor_thread = None
        self.preview_thread = None
        self.processed_files = []

        self.init_ui()
        self.setWindowTitle("Professional PDF Watermark Remover")
        self.setGeometry(100, 100, 1200, 800)

        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("PDF Watermark Removal Tool")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        settings_group = QGroupBox("Watermark Removal Settings")
        settings_layout = QHBoxLayout(settings_group)

        self.confidential_checkbox = QCheckBox("Remove 'CONFIDENTIAL' watermarks")
        self.confidential_checkbox.setChecked(True)
        settings_layout.addWidget(self.confidential_checkbox)

        self.generic_checkbox = QCheckBox("Auto-detect and remove repeating watermarks")
        self.generic_checkbox.setChecked(True)
        settings_layout.addWidget(self.generic_checkbox)

        layout.addWidget(settings_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel
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
        self.log_text.setMaximumHeight(220)
        self.log_text.setPlaceholderText("Processing log will appear here...")
        left_layout.addWidget(self.log_text)

        splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        preview_label = QLabel("PDF Preview")
        preview_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_layout.addWidget(preview_label)

        self.preview_widget = PDFPreviewWidget()
        right_layout.addWidget(self.preview_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])

        self.current_files = []

    # ----------------------------
    # File + (Conditional) Preview handling
    # ----------------------------
    def handle_files_dropped(self, files):
        """Handle newly selected files; generate auto processed preview only if a single file is selected."""
        # Cancel any in-flight preview
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.cancel()
            # do not join; avoid blocking

        self.current_files = files
        self.processed_files = []  # reset previous results
        self.log_text.clear()

        file_count = len(files)
        self.drop_zone.setText(f"📁 {file_count} PDF file{'s' if file_count != 1 else ''} selected\n\nReady to process!")
        self.start_button.setEnabled(True)

        self.log_text.append(f"Selected {file_count} PDF file(s):")
        for f in files[:10]:
            self.log_text.append(f"  • {os.path.basename(f)}")
        if file_count > 10:
            self.log_text.append(f"  ... and {file_count - 10} more files")

        self.status_label.setText(f"Ready to process {file_count} PDF file(s)")

        # --- Preview logic ---
        if file_count == 1:
            first_file = files[0]
            # Show immediate original preview; processed will be filled by preview thread
            self.preview_widget.update_preview(first_file, None)

            # Start background preview of processed version using current settings
            remove_confidential = self.confidential_checkbox.isChecked()
            remove_generic = self.generic_checkbox.isChecked()

            self.preview_thread = PreviewGeneratorThread(
                first_file,
                remove_confidential=remove_confidential,
                remove_generic=remove_generic
            )
            self.preview_thread.status_updated.connect(self.status_label.setText)
            self.preview_thread.error_occurred.connect(self.on_preview_error)
            self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
            self.preview_thread.start()

            self.log_text.append("🔍 Generating processed preview of the selected file…")
        else:
            # Disable preview if multiple docs are selected
            self.preview_widget.set_preview_disabled(
                "Preview disabled when multiple PDFs are selected."
            )

    def on_preview_error(self, file_path, error_message):
        filename = os.path.basename(file_path)
        self.log_text.append(f"❌ Preview error for {filename}: {error_message}")

    # ----------------------------
    # Processing controls
    # ----------------------------
    def start_processing(self):
        if not self.current_files:
            return

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        remove_confidential = self.confidential_checkbox.isChecked()
        remove_generic = self.generic_checkbox.isChecked()

        self.processor_thread = PDFProcessorThread(
            self.current_files, remove_confidential, remove_generic)

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

        # Always show the standard info dialog
        if success_count > 0:
            QMessageBox.information(
                self, "Processing Complete",
                f"Successfully processed {success_count} out of {total_count} PDF files.\n\n"
                f"Processed files are saved with '(No Watermarks)' prefix in the same directory."
            )

        # If more than one document processed, ask to open all at once
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


# ============================================================
# Entry point
# ============================================================
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
