# import os
# from typing import List
# from langchain_core.embeddings import Embeddings
# from huggingface_hub import InferenceClient

# class HFInferenceEmbeddings(Embeddings):
#     def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
#         self.client = InferenceClient(
#             provider="auto",
#             api_key=os.getenv("HF_TOKEN"),
#         )
#         self.model_name = model_name

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed a list of documents for Pinecone"""
#         res = self.client.feature_extraction(
#             texts,
#             model=self.model_name
#         )
#         return res.tolist() if hasattr(res, "tolist") else [list(e) for e in res]

#     def embed_query(self, text: str) -> List[float]:
#         """Embed a single query question"""
#         res = self.client.feature_extraction(
#             text,
#             model=self.model_name
#         )
#         vector = res.tolist() if hasattr(res, "tolist") else list(res)
        
#         if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
#             return vector[0]
#         return vector

import os
import requests
from typing import List, Optional
from langchain_core.embeddings import Embeddings

class CloudflareWorkersAIEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for Cloudflare Workers AI."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        model_name: str = "@cf/baai/bge-base-en-v1.5",
        batch_size: int = 64,
        timeout: int = 30,
    ):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.auth_token = auth_token or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout

        if not self.account_id:
            raise ValueError("Cloudflare Account ID must be provided or set via CLOUDFLARE_ACCOUNT_ID.")
        if not self.auth_token:
            raise ValueError("Cloudflare Auth Token must be provided or set via CLOUDFLARE_AUTH_TOKEN.")

        self.api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model_name}"
        )
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"text": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if not payload.get("success", False):
            errors = payload.get("errors", [])
            raise RuntimeError(f"Cloudflare Workers AI Error: {errors}")

        return payload.get("result", {}).get("data", [])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents in batches suitable for Pinecone."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            batch_result = self._call_api(chunk)
            embeddings.extend(batch_result)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single search query."""
        result = self._call_api([text])
        return result[0] if result else []