from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

mongo_url = os.getenv("MONGODB_CLUSTER_URL")
db_name = os.getenv("DATABASE_NAME", "DR_Bot")

client = None
db = None
chat_sessions = None
messages = None
patients = None

def get_db():
    global client, db, chat_sessions, messages, patients
    if not mongo_url:
        return None

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    needs_new_client = False
    if client is None:
        needs_new_client = True
    elif hasattr(client, "_io_loop") and client._io_loop and client._io_loop.is_closed():
        needs_new_client = True
    elif hasattr(client, "get_io_loop"):
        try:
            io_loop = client.get_io_loop()
            if io_loop and io_loop.is_closed():
                needs_new_client = True
        except Exception:
            pass

    if needs_new_client:
        try:
            client = AsyncIOMotorClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            db = client[db_name]
            chat_sessions = db.chat_sessions
            messages = db.messages
            patients = db.patients
        except Exception as e:
            print(f"[WARN] Failed to initialize MongoDB client: {e}")
            return None

    return db

def get_messages_collection():
    database = get_db()
    return database.messages if database is not None else None

def get_sessions_collection():
    database = get_db()
    return database.chat_sessions if database is not None else None

# Initialize default instances if possible
get_db()