"""
MongoDB database models and session management.
Stores interview sessions, Q&A history, and final reports.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Running without database persistence.")

from config import settings


class InterviewSession:
    """Represents a single interview session in the database."""

    COLLECTION = "interview_sessions"

    @staticmethod
    def create_document(
        role: str,
        candidate_name: str = "Unknown",
        has_resume: bool = False,
        has_jd: bool = False,
    ) -> Dict[str, Any]:
        """Create a new session document."""
        return {
            "_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "role": role,
            "candidate_name": candidate_name,
            "has_resume": has_resume,
            "has_jd": has_jd,
            "status": "in_progress",  # in_progress | completed | abandoned
            "qa_history": [],
            "current_question_index": 0,
            "final_report": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }


class DatabaseManager:
    """
    MongoDB database manager.
    Handles all CRUD operations for interview sessions.
    Falls back to in-memory storage if MongoDB is unavailable.
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._db: Optional[Any] = None
        self._in_memory: Dict[str, Dict[str, Any]] = {}  # Fallback storage
        self._use_mongo = False
        self._connect()

    def _connect(self):
        """Establish MongoDB connection."""
        if not MONGO_AVAILABLE:
            logger.warning("Using in-memory storage (pymongo not available)")
            return

        try:
            self._client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
            # Test connection
            self._client.admin.command("ping")
            self._db = self._client[settings.MONGODB_DB_NAME]
            self._use_mongo = True
            self._create_indexes()
            logger.info(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using in-memory storage.")
            self._use_mongo = False

    def _create_indexes(self):
        """Create database indexes for performance."""
        if not self._use_mongo:
            return
        try:
            sessions = self._db[InterviewSession.COLLECTION]
            sessions.create_index("session_id", unique=True)
            sessions.create_index("created_at")
            sessions.create_index("status")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")

    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new interview session."""
        session_id = session_data["session_id"]

        if self._use_mongo:
            self._db[InterviewSession.COLLECTION].insert_one(session_data)
        else:
            self._in_memory[session_id] = session_data.copy()

        logger.info(f"Session created: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        if self._use_mongo:
            doc = self._db[InterviewSession.COLLECTION].find_one(
                {"session_id": session_id},
                {"_id": 0}
            )
            return doc
        else:
            return self._in_memory.get(session_id)

    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update fields in a session."""
        update_data["updated_at"] = datetime.utcnow().isoformat()

        if self._use_mongo:
            result = self._db[InterviewSession.COLLECTION].update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        else:
            if session_id in self._in_memory:
                self._in_memory[session_id].update(update_data)
                return True
            return False

    def append_qa(self, session_id: str, qa_item: Dict[str, Any]) -> bool:
        """Append a Q&A item to session history."""
        if self._use_mongo:
            result = self._db[InterviewSession.COLLECTION].update_one(
                {"session_id": session_id},
                {
                    "$push": {"qa_history": qa_item},
                    "$set": {"updated_at": datetime.utcnow().isoformat()},
                    "$inc": {"current_question_index": 1},
                }
            )
            return result.modified_count > 0
        else:
            session = self._in_memory.get(session_id)
            if session:
                session["qa_history"].append(qa_item)
                session["current_question_index"] += 1
                session["updated_at"] = datetime.utcnow().isoformat()
                return True
            return False

    def save_report(self, session_id: str, report: Dict[str, Any]) -> bool:
        """Save the final HR report to session."""
        update = {
            "final_report": report,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
        }
        success = self.update_session(session_id, update)
        if success:
            logger.info(f"Report saved for session: {session_id}")
        return success

    def list_sessions(
        self,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List recent sessions."""
        if self._use_mongo:
            query = {}
            if status:
                query["status"] = status
            cursor = (
                self._db[InterviewSession.COLLECTION]
                .find(query, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return list(cursor)
        else:
            sessions = list(self._in_memory.values())
            if status:
                sessions = [s for s in sessions if s.get("status") == status]
            sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if self._use_mongo:
            result = self._db[InterviewSession.COLLECTION].delete_one(
                {"session_id": session_id}
            )
            return result.deleted_count > 0
        else:
            if session_id in self._in_memory:
                del self._in_memory[session_id]
                return True
            return False

    def close(self):
        """Close database connection."""
        if self._client:
            self._client.close()


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager