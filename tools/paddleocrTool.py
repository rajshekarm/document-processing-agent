import os
from pathlib import Path
from paddleocr import PaddleOCR
from dotenv import load_dotenv


os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
# Load environment variables from .env
_ = load_dotenv(override=True)

print(Path.home())


# Initialize English OCR model
ocr = PaddleOCR(lang="en", enable_mkldnn=False)

# # We must download both models for PaddleOCR. Be aware that the downloads will take a moment. 
# # Subsequently, the models will load from cached files in the `.paddlex` folder.


image_path = '../receipt.jpeg'
# img = Image.open(image_path)

res = ocr.predict(image_path)

page = res[0]
texts = page['rec_texts']

for text in texts:
    print(text)