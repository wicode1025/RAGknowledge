"""
Nougat PDF Parser
Parses PDF documents to Markdown format, preserving pseudocode and mathematical formulas.
Note: This is a simple fallback parser. For full Nougat support,
      install the nougat-ocr package separately.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import logging

from config import Config

logger = logging.getLogger(__name__)


class NougatParser:
    """PDF parser using simple extraction (fallback when Nougat unavailable)"""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize parser

        Args:
            model_name: Nougat model name (not used in fallback)
            device: Device (not used in fallback)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name or Config.NOUGAT_MODEL
        logger.info("Using simple PDF parser (Nougat not available)")

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> str:
        """
        Parse a PDF file to Markdown

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save output markdown (optional)
            start_page: Starting page (0-indexed)
            end_page: Ending page (exclusive, None for all pages)

        Returns:
            Combined markdown text from all pages
        """
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path}")

        # Open PDF
        doc = fitz.open(str(pdf_path))

        if end_page is None:
            end_page = len(doc)

        markdown_parts = []

        # Process each page
        for page_num in range(start_page, min(end_page, len(doc))):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")

            # Extract text
            page = doc[page_num]
            text = page.get_text()

            # Basic formatting
            if text.strip():
                markdown_parts.append(f"## Page {page_num + 1}\n\n{text}")

        doc.close()

        # Combine all markdown
        combined_md = "\n\n".join(markdown_parts)

        # Save to file if output_dir specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{pdf_path.stem}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_md)
            logger.info(f"Saved markdown to: {output_path}")

        return combined_md

    def parse_pdf_directory(
        self,
        pdf_dir: Union[str, Path],
        output_dir: Optional[Path] = None,
        extensions: tuple = ('.pdf',)
    ) -> List[str]:
        """
        Parse all PDFs in a directory

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save output markdown files
            extensions: File extensions to process

        Returns:
            List of paths to generated markdown files
        """
        pdf_dir = Path(pdf_dir)
        output_dir = output_dir or Config.PROCESSED_DIR

        markdown_files = []

        # Find all PDF files
        pdf_files = []
        for ext in extensions:
            pdf_files.extend(pdf_dir.glob(f"*{ext}"))
        for ext in extensions:
            pdf_files.extend(pdf_dir.rglob(f"*{ext}"))

        # Remove duplicates
        pdf_files = list(set(pdf_files))

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            try:
                md_path = self.parse_pdf(
                    pdf_path,
                    output_dir=output_dir
                )
                markdown_files.append(md_path)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")

        return markdown_files


class SimpleMarkdownParser:
    """
    Simple fallback parser that extracts text from PDFs without Nougat.
    Uses pdfplumber + OCR for better text extraction including Chinese.
    """

    def __init__(self):
        """Initialize simple parser"""
        import os
        # Set Tesseract path
        tesseract_path = r"D:\APP\OCR"
        if tesseract_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] += os.pathsep + tesseract_path

        # Set Poppler path
        poppler_path = r"D:\APP\poppler\Release-25.12.0-0\poppler-25.12.0\Library\bin"
        if os.path.exists(poppler_path) and poppler_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] += os.pathsep + poppler_path

        # 设置环境变量供pdf2image使用
        import subprocess
        import shutil
        if shutil.which('pdftoppm') is None and os.path.exists(poppler_path):
            os.environ['PATH'] = poppler_path + os.pathsep + os.environ.get('PATH', '')

        self.tesseract_path = tesseract_path

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        use_ocr: bool = False,
        max_pages: int = 50
    ) -> str:
        """
        Parse PDF to plain text

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save output
            use_ocr: Enable OCR for scanned pages
            max_pages: Max pages to process

        Returns:
            Extracted text
        """
        import pdfplumber

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text_parts = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            pages_to_process = min(max_pages, total_pages)
            logger.info(f"Processing {pdf_path.name} ({total_pages} pages, will process {pages_to_process})")

            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                text = page.extract_text()

                # 如果提取的文字太少且启用OCR，尝试OCR
                if use_ocr and (not text or len(text.strip()) < 100):
                    try:
                        from pdf2image import convert_from_path
                        import pytesseract
                        images = convert_from_path(str(pdf_path), first_page=page_num+1, last_page=page_num+1, dpi=200)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0], lang='chi_sim+eng')
                            if ocr_text and len(ocr_text.strip()) > 50:
                                text = ocr_text
                    except Exception as e:
                        logger.warning(f"OCR failed: {e}")

                if text:
                    text_parts.append(f"## Page {page_num + 1}\n\n{text}")

            if total_pages > pages_to_process:
                text_parts.append(f"\n\n... (共 {total_pages} 页) ...")

        combined_text = "\n\n".join(text_parts)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{pdf_path.stem}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_text)

        return combined_text


if __name__ == "__main__":
    # Test parser
    logging.basicConfig(level=logging.INFO)

    parser = NougatParser()

    # Example: Parse a PDF if exists
    pdfs = list(Config.PDFS_DIR.glob("*.pdf"))
    if pdfs:
        result = parser.parse_pdf(pdfs[0], output_dir=Config.PROCESSED_DIR)
        print(f"Parsed {len(result)} characters")
    else:
        print("No PDFs found in data directory")
