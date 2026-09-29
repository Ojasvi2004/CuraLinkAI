from typing import Optional, Dict, Any
import asyncio
import traceback
import logging

from components.retrievers.temp_retrieve import rag_chain_lcel, vector_store
from database.mongoDB.conversation import (
    save_message,
    get_recent_history,
    history_to_prompt
)

logger = logging.getLogger(__name__)

gemini_pipeline = rag_chain_lcel(vector_store)

async def rag_main_api_function(query: dict):
    try:
        # Support top-level 'question', nested under 'query', or query string
        if "question" in query:
            q_text = str(query["question"]).strip()
            patient_data = query.get("patient") or {}
            session_id = query.get("session_id") or "default_session"
        elif "query" in query and isinstance(query["query"], dict) and "question" in query["query"]:
            q_text = str(query["query"]["question"]).strip()
            patient_data = query["query"].get("patient") or {}
            session_id = query["query"].get("session_id") or "default_session"
        elif "query" in query and isinstance(query["query"], str):
            q_text = query["query"].strip()
            patient_data = query.get("patient") or {}
            session_id = query.get("session_id") or "default_session"
        else:
            raise ValueError("Missing 'question' in payload")

        if not q_text:
            raise ValueError("Question cannot be empty")

        # 1. Fetch recent conversation history
        try:
            history = await get_recent_history(session_id=session_id)
            history_text = history_to_prompt(history)
        except Exception as e:
            logger.warning(f"Error fetching chat history: {e}")
            history_text = ""

        # 2. Persist user's question to MongoDB
        try:
            await save_message(
                session_id=session_id,
                role="user",
                content=q_text
            )
        except Exception as e:
            logger.warning(f"Error saving user message: {e}")

        # 3. Execute LCEL pipeline in worker thread to prevent event loop blocking
        response = await asyncio.to_thread(
            gemini_pipeline.invoke,
            {
                "question": q_text,
                "patient": patient_data,
                "chat_history": history_text
            }
        )

        # 4. Safely extract answer string and sources
        if isinstance(response, dict):
            answer = response.get("answer") or response.get("result") or response.get("response") or response.get("raw_output") or ""
            sources = response.get("sources", [])
        else:
            answer = str(response)
            sources = []

        # 5. Persist assistant's response to MongoDB
        try:
            await save_message(
                session_id=session_id,
                role="assistant",
                content=answer
            )
        except Exception as e:
            logger.warning(f"Error saving assistant message: {e}")

        return {
            "success": True,
            "result": answer,
            "sources": sources,
            "data": response if isinstance(response, dict) else {"answer": answer}
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": f"{str(e)}, error in rag.py"
        }
