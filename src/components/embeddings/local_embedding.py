import os
import json
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from configurations.config import EMBEDDING_DIRECTORY, META_DATA_DIRECTORY, CHUNKS_DIRECTORY

load_dotenv()

client = None

def get_client():
    global client
    if client is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables.")
        client = InferenceClient(api_key=hf_token)
    return client


def _load_chunks(directory):
    chunks = []
    metadata = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return chunks, metadata

    files = [f for f in os.listdir(directory) if f.lower().endswith(".json")]
    if not files:
        print("No JSON files found in directory")
        return chunks, metadata

    for fileName in tqdm(files, desc="Loading Chunks"):
        filePath = os.path.join(directory, fileName)
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, chunk in enumerate(data):
            chunks.append(chunk)
            metadata.append({
                "source_file": fileName,
                "chunk_number": i,
                "chunk_text": chunk
            })

    print(f"[INFO] Total chunks loaded: {len(chunks)}")
    return chunks, metadata


def embed_chunks(directory):
    chunks, metadata = _load_chunks(directory)
    if len(chunks) == 0:
        print("No chunks to embed.")
        return

    hf_client = get_client()
    batch_size = 24
    all_embeddings = []

    print(f"[INFO] Embedding {len(chunks)} chunks via Hugging Face Cloud API...")
    
  
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Batches"):
        batch = chunks[i:i + batch_size]
        res = hf_client.feature_extraction(
            batch,
            model="BAAI/bge-base-en-v1.5"
        )
        batch_emb = np.array(res)
        

        norm = np.linalg.norm(batch_emb, axis=1, keepdims=True)
        norm_emb = batch_emb / np.clip(norm, a_min=1e-12, a_max=None)
        all_embeddings.append(norm_emb)

    
    embeddings = np.vstack(all_embeddings)

 
    np.save(EMBEDDING_DIRECTORY, embeddings)
    print(f"✅ Embeddings saved to {EMBEDDING_DIRECTORY} (Shape: {embeddings.shape})")

   
    with open(META_DATA_DIRECTORY, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata saved to {META_DATA_DIRECTORY}")
    print("🎉 Embedding process completed successfully.")


if __name__ == "__main__":
    embed_chunks(CHUNKS_DIRECTORY)