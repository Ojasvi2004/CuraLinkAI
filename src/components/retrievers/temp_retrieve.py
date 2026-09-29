import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

if "HF_HOME" not in os.environ and os.path.exists("E:/hf_cache"):
    os.environ["HF_HOME"] = "E:/hf_cache"
else :
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# from components.vector_store.pinecone_store import RealEmbeddings
from components.embeddings.remote_embedding import HFInferenceEmbeddings
from components.prompts.query_prompt import prompt1

def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "gemini" and gemini_api_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=gemini_api_key,
            temperature=0.2
        )

    if groq_api_key:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model=model_name,
            api_key=groq_api_key,
            temperature=0.2
        )

    if gemini_api_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=gemini_api_key,
            temperature=0.2
        )

    raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY found in environment variables.")

llm = get_llm()

embeddings = HFInferenceEmbeddings()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "medical-rag-bge"),
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)

def rag_chain_lcel(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": int(os.getenv("RAG_TOP_K", 10))})

    def retrieve_and_format(x):
        query = x.get("query", "").strip()
        docs = retriever.invoke(query) if query else []

        formatted_docs = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source_file", "Medical Knowledge Reference")
            chunk_num = d.metadata.get("chunk_number", i)
            formatted_docs.append(f"[Document {i} | Source: {source} | Chunk: {chunk_num}]\n{d.page_content}")

        context_str = "\n\n".join(formatted_docs) if formatted_docs else "No specific medical reference documents found."

        return {
            "query": query,
            "context": context_str,
            "patient": x.get("patient") or {},
            "chat_history": x.get("chat_history", ""),
            "docs": docs
        }

    def llm_with_patient(x):
        final_prompt = prompt1.format(
            context=x["context"],
            question=x["query"],
            patient=json.dumps(x.get("patient") or {}, indent=2),
            chat_history=x.get("chat_history", "")
        )

        response = llm.invoke(final_prompt)
        content = response.content

        if isinstance(content, list):
            raw_output = "".join(
                block.get("text", "") if isinstance(block, dict) else (block.text if hasattr(block, "text") else str(block))
                for block in content
            )
        else:
            raw_output = str(content)

        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        raw_output = raw_output.strip()

        sources = []
        seen = set()
        for d in x.get("docs", []):
            sf = d.metadata.get("source_file", "Medical Knowledge Reference")
            cn = d.metadata.get("chunk_number", None)
            key = (sf, cn)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source_file": sf,
                    "chunk_number": cn
                })

        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                if "answer" not in parsed:
                    parsed["answer"] = parsed.get("result") or parsed.get("response") or raw_output
                parsed["sources"] = sources
                return parsed
        except Exception:
            pass

        return {
            "answer": raw_output,
            "confidence": "high" if len(raw_output) > 50 else "unknown",
            "warnings": [],
            "sources": sources,
            "raw_output": raw_output
        }

    full_pipeline = RunnableSequence(
        RunnableLambda(lambda x: {
            "query": x.get("question") or x.get("query", ""),
            "patient": x.get("patient") or {},
            "chat_history": x.get("chat_history", "")
        }),
        RunnableLambda(retrieve_and_format),
        RunnableLambda(llm_with_patient)
    )

    return full_pipeline

