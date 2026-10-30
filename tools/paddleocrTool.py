import os
from pathlib import Path
from paddleocr import PaddleOCR
from dotenv import load_dotenv
from langchain.tools import tool
import base64
from typing import List, Dict, Any



@tool
def paddle_ocr_document(image_path:str) -> List[Dict[str, int]]:
    
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    # Load environment variables from .env
    _ = load_dotenv(override=True)

    # Initialize English OCR model with CPU infernce disable
    ocr = PaddleOCR(lang="en", enable_mkldnn=False)

    # # We must download both models for PaddleOCR. Be aware that the downloads will take a moment. 
    # # Subsequently, the models will load from cached files in the `.paddlex` folder.

    res = ocr.predict(image_path)

    page = res[0]
    texts = page['rec_texts']
    texts = page['rec_texts'] # recognized text strings
    scores = page['rec_scores'] # confidence scores for each text line
    boxes  = page['rec_polys'] #  bounding box coordinates

    extracted_items = []
    for text, box, score in zip(texts, boxes, scores):
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        bbox = [min(x_coords), min(y_coords), max(x_coords), 
                max(y_coords)]
        
        item = {
            'text': text,
            'bbox': bbox,
        }
        if score is not None:
            item['confidence'] = score
            
        extracted_items.append(item)
    
    return extracted_items


# PaddleOCR preprocesses images automatically—correcting rotation, deskewing, and removing noise.


# 
    return [{"text" : 1, "text2" : 2}, {"text" : 1, "text2" : 2}]