
from pathlib import Path
from io import BytesIO

import pymupdf
import pytesseract

from PIL import Image
from langchain.tools import tool


@tool
def ocr_read_document(file_path: str) -> str:
    """
    Extract text from a PDF or image document.

    Use this tool when the user asks to read,
    extract, or analyze text from a document.

    Args:
        file_path: Local path to a PDF, PNG, JPG,
                   or JPEG document.

    Returns:
        Extracted text from the document.
    """

    path = Path(file_path).expanduser()

    # Check whether file exists
    if not path.is_file():
        return f"Error: File not found: {file_path}"

    extension = path.suffix.lower()

    try:
        # Handle PDF files
        if extension == ".pdf":

            extracted_pages = []

            with pymupdf.open(path) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    # First try normal text extraction
                    text = page.get_text("text").strip()

                    # If no text, use OCR
                    if not text:

                        pix = page.get_pixmap(dpi=300)

                        image = Image.open(
                            BytesIO(pix.tobytes("png"))
                        )

                        text = pytesseract.image_to_string(
                            image,
                            lang="eng"
                        ).strip()

                    extracted_pages.append(
                        f"--- Page {page_number} ---\n{text}"
                    )

            result = "\n\n".join(extracted_pages)

        # Handle image files
        elif extension in [".png", ".jpg", ".jpeg"]:

            with Image.open(path) as image:
                result = pytesseract.image_to_string(
                    image,
                    lang="eng"
                ).strip()

        else:
            return "Error: Unsupported file format."

        if not result.strip():
            return "No readable text found in the document."

        return result

    except Exception as e:
        return f"OCR processing failed: {str(e)}"
