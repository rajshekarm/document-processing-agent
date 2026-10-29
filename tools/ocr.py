from langchain.tools import tool
import pytesseract
from PIL import Image 
# PIL Loads and manipulate images

# The @tool decorator from LangChain transforms a function into a tool for use by agents.
@tool
def ocr_read_document(image_path : str) -> str:
    """Reads an image from the given path and returns extracted text using OCR"""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        return f"Error reading image {e}"

