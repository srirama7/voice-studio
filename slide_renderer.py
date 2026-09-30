"""
Slide Renderer Module for Voice Studio.

Automates rendering of visual slide images to 1920x1080 PNG format:
1. PyMuPDF (`fitz`) vector PDF page rendering (scaled to 1920x1080).
2. Headless LibreOffice PPTX-to-PDF conversion + PyMuPDF high-res rendering.
3. PIL Visual Card Renderer fallback generating styled dark/light presentation cards.
"""

import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Union, Any

from PIL import Image, ImageDraw, ImageFont

from config import Config, default_config
from document_parser import ParsedDocument, SlideData, DocumentParser

# Try PyMuPDF (pymupdf / fitz)
try:
    import pymupdf as fitz
    _HAS_FITZ = True
except ImportError:
    try:
        import fitz
        _HAS_FITZ = True
    except ImportError:
        _HAS_FITZ = False


class SlideRenderer:
    """Automated Slide Visual Renderer generating 1920x1080 PNG slide images."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or default_config
        self.width = self.config.slide.target_width
        self.height = self.config.slide.target_height

    def render_document(
        self,
        parsed_doc: ParsedDocument,
        output_dir: Union[str, Path]
    ) -> List[Path]:
        """
        Render all slides of a ParsedDocument into 1920x1080 PNG files in output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_file = Path(parsed_doc.file_path) if parsed_doc.file_path else None

        if source_file and source_file.exists():
            ext = source_file.suffix.lower()
            if ext == ".pdf" and _HAS_FITZ:
                return self.render_pdf(source_file, output_dir)
            elif ext == ".pptx":
                return self.render_pptx(source_file, output_dir, parsed_doc=parsed_doc)

        # Fallback to visual PIL card renderer for all slides
        rendered_images: List[Path] = []
        for idx, slide in enumerate(parsed_doc.slides, start=1):
            out_img = output_dir / f"slide_{idx:02d}.png"
            self.render_slide_card(slide, out_img, total_slides=parsed_doc.total_slides)
            rendered_images.append(out_img)

        return rendered_images

    def render_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> List[Path]:
        """Render vector PDF pages directly to 1920x1080 PNG images via PyMuPDF (fitz)."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not _HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required to render PDF slides. Install via `pip install pymupdf`.")

        doc = fitz.open(str(pdf_path))
        rendered_images: List[Path] = []

        for idx, page in enumerate(doc, start=1):
            page_rect = page.rect
            scale_x = float(self.width) / page_rect.width
            scale_y = float(self.height) / page_rect.height
            mat = fitz.Matrix(scale_x, scale_y)

            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = output_dir / f"slide_{idx:02d}.png"
            pix.save(str(out_path))
            rendered_images.append(out_path)

        doc.close()
        return rendered_images

    def render_pptx(
        self,
        pptx_path: Union[str, Path],
        output_dir: Union[str, Path],
        parsed_doc: Optional[ParsedDocument] = None
    ) -> List[Path]:
        """
        Render PPTX slides.
        Attempts LibreOffice PPTX->PDF conversion + PyMuPDF rendering first.
        Falls back to PIL visual card rendering if LibreOffice is unavailable.
        """
        pptx_path = Path(pptx_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Look for LibreOffice executable
        soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
        if _HAS_FITZ and soffice_bin:
            try:
                temp_pdf_dir = output_dir / "temp_pdf"
                temp_pdf_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    soffice_bin,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    str(pptx_path),
                    "--outdir",
                    str(temp_pdf_dir)
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                converted_pdf = temp_pdf_dir / f"{pptx_path.stem}.pdf"
                if res.returncode == 0 and converted_pdf.exists():
                    images = self.render_pdf(converted_pdf, output_dir)
                    shutil.rmtree(temp_pdf_dir, ignore_errors=True)
                    return images
            except Exception:
                pass  # Fallback to PIL renderer below

        # Fallback: PIL Visual Card Renderer
        if parsed_doc is None:
            parsed_doc = DocumentParser.parse_pptx(pptx_path)

        rendered_images: List[Path] = []
        for idx, slide in enumerate(parsed_doc.slides, start=1):
            out_img = output_dir / f"slide_{idx:02d}.png"
            self.render_slide_card(slide, out_img, total_slides=parsed_doc.total_slides)
            rendered_images.append(out_img)

        return rendered_images

    def render_slide_card(
        self,
        slide: SlideData,
        output_path: Union[str, Path],
        total_slides: int = 1
    ) -> Path:
        """
        Generate a styled 1920x1080 presentation visual card using PIL (Pillow).
        Supports professional dark theme, slide title, body text wrapping, and slide badge.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Background color: Dark Navy #0f172a
        bg_color = (15, 23, 42)
        card_bg = (30, 41, 59)
        accent_color = (59, 130, 246)  # Blue accent
        title_color = (248, 250, 252)
        body_color = (203, 213, 225)
        badge_color = (71, 85, 105)

        img = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw main content card rectangle
        card_rect = [80, 80, self.width - 80, self.height - 80]
        draw.rounded_rectangle(card_rect, radius=20, fill=card_bg)

        # Draw top accent bar
        draw.rounded_rectangle([80, 80, self.width - 80, 96], radius=8, fill=accent_color)

        # Load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 52)
            body_font = ImageFont.truetype("arial.ttf", 32)
            badge_font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            badge_font = ImageFont.load_default()

        # Draw Slide Badge (e.g. "SLIDE 01 / 05")
        badge_text = f"SLIDE {slide.slide_index:02d} / {total_slides:02d}"
        draw.text((120, 130), badge_text, font=badge_font, fill=accent_color)

        # Draw Title
        title_text = slide.title or f"Slide {slide.slide_index}"
        draw.text((120, 180), title_text, font=title_font, fill=title_color)

        # Draw Title Divider
        draw.line([(120, 260), (self.width - 120, 260)], fill=badge_color, width=2)

        # Wrap and Draw Body Text
        body_text = slide.body_text or slide.raw_text or ""
        lines = self._wrap_text(body_text, body_font, max_width=self.width - 240, draw=draw)

        y_offset = 300
        for line in lines[:16]:  # Limit lines to fit card
            draw.text((120, y_offset), line, font=body_font, fill=body_color)
            y_offset += 42

        img.save(output_path, format="PNG")
        return output_path

    def _wrap_text(
        self,
        text: str,
        font: Any,
        max_width: int,
        draw: ImageDraw.ImageDraw
    ) -> List[str]:
        """Wrap text lines to fit max_width pixel constraint."""
        wrapped_lines = []
        paragraphs = text.splitlines()

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            words = para.split()
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]

                if line_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word

            if current_line:
                wrapped_lines.append(current_line)

        return wrapped_lines
