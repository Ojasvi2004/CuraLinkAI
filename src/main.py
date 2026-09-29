import os
import json
import uvicorn
import traceback
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

if "HF_HOME" not in os.environ and os.path.exists("E:/hf_cache"):
    os.environ["HF_HOME"] = "E:/hf_cache"

from api.v1.endpoints.ocr_query import ocr_main_api_function
from api.v1.endpoints.client_query import rag_main_api_function


app = FastAPI(
    title="Dr.Bot Medical Assistant API",
    version="1.0.0",
    description="Production-ready Medical RAG and OCR API"
)

cors_env = os.getenv("CORS_ORIGINS", "")
origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()] if cors_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_credentials=True if "*" not in origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OCRRequest(BaseModel):
    image_url: str

class QueryRequest(BaseModel):
    question: str
    patient: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = "default_session"

def clean_json_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Dr.Bot API"}

@app.post("/ocr")
def get_ocr_data(request: OCRRequest):
    try:
        result = ocr_main_api_function(request.image_url)

        if isinstance(result, dict):
            return {"success": True, "data": result}

        cleaned_result = clean_json_markdown(result)
        try:
            json_result = json.loads(cleaned_result)
        except json.JSONDecodeError:
            json_result = result

        return {"success": True, "data": json_result}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/askDrBot")
async def ask_dr_bot(request: QueryRequest):
    try:
        result = await rag_main_api_function(request.model_dump())
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": f"{str(e)}, error in main.py"
        }

if __name__ == "__main__":

    port = int(os.getenv("PORT", 8000))

    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)