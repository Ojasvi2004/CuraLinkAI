from .mongo import get_messages_collection
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

async def save_message(
    session_id: str,
    role: str,
    content: str
):
    coll = get_messages_collection()
    if coll is None or not session_id or not content:
        return None

    try:
        result = await coll.insert_one({
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        })
        return result
    except Exception as e:
        logger.warning(f"Failed to save message to MongoDB: {e}")
        return None

async def get_recent_history(
    session_id: str,
    limit: int = 10
):
    coll = get_messages_collection()
    if coll is None or not session_id:
        return []

    try:
        cursor = (
            coll.find(
                {"session_id": session_id}
            )
            .sort("timestamp", -1)
            .limit(limit)
        )

        history = await cursor.to_list(length=limit)
        history.reverse()
        return history
    except Exception as e:
        logger.warning(f"Failed to retrieve history from MongoDB: {e}")
        return []

def history_to_prompt(history):
    if not history:
        return ""

    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)