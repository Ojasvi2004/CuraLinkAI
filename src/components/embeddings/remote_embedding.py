import os
from typing import List
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.client = InferenceClient(
            provider="auto",
            api_key=os.getenv("HF_TOKEN"),
        )
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for Pinecone"""
        res = self.client.feature_extraction(
            texts,
            model=self.model_name
        )
        return res.tolist() if hasattr(res, "tolist") else [list(e) for e in res]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query question"""
        res = self.client.feature_extraction(
            text,
            model=self.model_name
        )
        vector = res.tolist() if hasattr(res, "tolist") else list(res)
        
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            return vector[0]
        return vector