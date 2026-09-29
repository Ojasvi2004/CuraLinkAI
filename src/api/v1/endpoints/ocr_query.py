from pydantic import BaseModel
from typing import Optional
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from components.generators.ocr_json_generator import ocr_main_pipeline

class QueryRequest(BaseModel):
    query: str
    image_url: Optional[str] = None

def load_image_from_url(url: str):
   
    response = requests.get(url)
    response.raise_for_status() 
    image = Image.open(BytesIO(response.content))
    return image

def ocr_main_api_function(url: str):
    print("\n========== API OCR START ==========")

    image = load_image_from_url(url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        temp_path = tmp.name

    image.save(temp_path)

    print("Temporary image:", temp_path)
    print("Image mode:", image.mode)
    print("Image size:", image.size)

    try:
        result = ocr_main_pipeline.invoke(temp_path)

        print("\n========== PIPELINE RESULT ==========")
        print("RESULT TYPE:", type(result))
        print("RESULT:", repr(result))

        return result

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)