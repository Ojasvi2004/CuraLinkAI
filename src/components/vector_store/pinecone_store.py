import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

if "HF_HOME" not in os.environ and os.path.exists("E:/hf_cache"):
    os.environ["HF_HOME"] = "E:/hf_cache"

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from components.embeddings.local_embedding import _load_chunks
from configurations.config import EMBEDDING_DIRECTORY, META_DATA_DIRECTORY, CHUNKS_DIRECTORY

_shared_embedding_model = None

class RealEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        global _shared_embedding_model
        if _shared_embedding_model is None:
            _shared_embedding_model = SentenceTransformer(model_name)
        self.model = _shared_embedding_model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None
    return Pinecone(api_key=api_key)

pc = get_pinecone_client()


def build_pine_cone():
    index_name = "medical-rag-bge"
    embeddings=np.load(EMBEDDING_DIRECTORY)

    if not pc.has_index(index_name):
        pc.create_index(
                name=index_name,
                dimension=embeddings.shape[1], 
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index=pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        print("Index already populated. Skipping upsert.")
        return
    records=[]
    chunks,meta_data=_load_chunks(CHUNKS_DIRECTORY)
    
    for i,meta in enumerate(meta_data):
       records.append({
            "id": f"chunk-{i}",
            "values": embeddings[i].tolist(),  
             "metadata": {
            "text": meta_data[i]["chunk_text"],
            "source_file": meta_data[i]["source_file"],
            "chunk_number": meta_data[i]["chunk_number"]
        }
})
    index.upsert(records,batch_size=100)
    print(f"Upserted {len(records)} into Pinecone")
    
    



if __name__ == "__main__":
    
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"API Key found: {'Yes' if os.getenv('PINECONE_API_KEY') else 'No'}")
    build_pine_cone()