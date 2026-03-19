"""
OCR-enabled PDF Parser
Uses pytesseract for scanned Chinese PDFs
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class OCRPDFParser:
    """
    PDF parser with OCR support for scanned documents.
    Falls back to pdfplumber for text-based PDFs.
    """

    def __init__(self, use_ocr_threshold: int = 500):
        """
        Initialize parser

        Args:
            use_ocr_threshold: If extracted text is less than this, use OCR
        """
        self.use_ocr_threshold = use_ocr_threshold
        self._has_ocr = self._check_ocr_available()

    def _check_ocr_available(self) -> bool:
        """Check if OCR is available"""
        try:
            import pytesseract
            import pdf2image
            return True
        except ImportError:
            return False

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        use_ocr: bool = False,
        first_n_pages: int = 50  # 只OCR前N页以节省时间
    ) -> str:
        """
        Parse PDF with optional OCR

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save output
            use_ocr: Force OCR usage
            first_n_pages: Number of pages to process with OCR

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
            logger.info(f"Processing {total_pages} pages from {pdf_path.name}")

            # Determine if we need OCR
            pages_to_extract = min(first_n_pages, total_pages)

            for i, page in enumerate(pdf.pages[:pages_to_extract]):
                # Try text extraction first
                text = page.extract_text()

                # Check if text is too short (might be scanned)
                if text and len(text.strip()) < self.use_ocr_threshold:
                    if use_ocr or self._has_ocr:
                        logger.info(f"Page {i+1} has little text, trying OCR...")
                        try:
                            # Try OCR
                            import pytesseract
                            from pdf2image import convert_from_path

                            images = convert_from_path(str(pdf_path), first_page=i+1, last_page=i+1)
                            if images:
                                ocr_text = pytesseract.image_to_string(images[0], lang='chi_sim+eng')
                                if ocr_text.strip():
                                    text = ocr_text
                        except Exception as e:
                            logger.warning(f"OCR failed for page {i+1}: {e}")

                if text:
                    text_parts.append(f"## Page {i + 1}\n\n{text}")

            if pages_to_extract < total_pages:
                text_parts.append(f"\n\n... (共 {total_pages} 页, 仅处理前 {pages_to_extract} 页) ...")

        combined_text = "\n\n".join(text_parts)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{pdf_path.stem}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_text)

        return combined_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = OCRPDFParser()

    # Test with a Chinese PDF
    pdfs = list(Path('book').glob('*.pdf'))
    if pdfs:
        result = parser.parse_pdf(pdfs[0], output_dir=Path('data/processed'))
        print(f"Extracted {len(result)} characters")
