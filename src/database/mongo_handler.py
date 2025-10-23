"""MongoDB handler for chat history persistence."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)


class MongoHandler:
    """Handles MongoDB operations for chat history."""
    
    def __init__(self, uri: str, database: str, collection: str):
        """
        Initialize MongoDB handler.
        
        Args:
            uri: MongoDB connection URI
            database: Database name
            collection: Collection name
        """
        self.uri = uri
        self.database_name = database
        self.collection_name = collection
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            await self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self._connected = True
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self._connected
    
    async def setup_collection(self) -> Dict[str, Any]:
        """
        Create collection with proper schema and indexes.
        
        Returns:
            Status dictionary with success/error information
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return {"success": False, "error": "Not connected to MongoDB"}
        
        try:
            # Create collection if it doesn't exist
            collections = await self.db.list_collection_names()
            if self.collection_name not in collections:
                await self.db.create_collection(self.collection_name)
                logger.info(f"Created collection: {self.collection_name}")
            
            # Create indexes for efficient querying
            await self.collection.create_index([("session_id", ASCENDING)])
            await self.collection.create_index([("timestamp", DESCENDING)])
            await self.collection.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
            
            # Optionally set up schema validation
            validator = {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["session_id", "timestamp", "role", "content"],
                    "properties": {
                        "session_id": {
                            "bsonType": "string",
                            "description": "Session identifier"
                        },
                        "timestamp": {
                            "bsonType": "date",
                            "description": "Message timestamp"
                        },
                        "role": {
                            "enum": ["user", "assistant", "system"],
                            "description": "Message role"
                        },
                        "content": {
                            "bsonType": "string",
                            "description": "Message content"
                        },
                        "tools_used": {
                            "bsonType": "array",
                            "description": "List of tools used in this message"
                        },
                        "metadata": {
                            "bsonType": "object",
                            "description": "Additional metadata"
                        }
                    }
                }
            }
            
            # Update collection with validator
            await self.db.command({
                "collMod": self.collection_name,
                "validator": validator,
                "validationLevel": "moderate"
            })
            
            logger.info(f"Collection setup complete: {self.collection_name}")
            return {
                "success": True,
                "message": f"Collection '{self.collection_name}' created successfully with indexes and schema validation"
            }
            
        except OperationFailure as e:
            logger.error(f"Failed to setup collection: {e}")
            return {"success": False, "error": str(e)}
    
    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a chat message to MongoDB.
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            tools_used: List of tools used (optional)
            metadata: Additional metadata (optional)
        
        Returns:
            True if save successful, False otherwise
        """
        if not self._connected:
            logger.warning("Not connected to MongoDB, cannot save message")
            return False
        
        try:
            document = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "role": role,
                "content": content,
                "tools_used": tools_used or [],
                "metadata": metadata or {}
            }
            
            await self.collection.insert_one(document)
            logger.debug(f"Saved message for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve (optional)
        
        Returns:
            List of message documents
        """
        if not self._connected:
            logger.warning("Not connected to MongoDB, cannot retrieve history")
            return []
        
        try:
            query = {"session_id": session_id}
            cursor = self.collection.find(query).sort("timestamp", ASCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            messages = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for msg in messages:
                msg["_id"] = str(msg["_id"])
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []
    
    async def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent chat sessions.
        
        Args:
            limit: Maximum number of sessions to retrieve
        
        Returns:
            List of session summaries
        """
        if not self._connected:
            logger.warning("Not connected to MongoDB, cannot list sessions")
            return []
        
        try:
            # Aggregate to get unique sessions with latest message timestamp
            pipeline = [
                {
                    "$sort": {"timestamp": -1}
                },
                {
                    "$group": {
                        "_id": "$session_id",
                        "last_message": {"$first": "$timestamp"},
                        "message_count": {"$sum": 1},
                        "latest_content": {"$first": "$content"}
                    }
                },
                {
                    "$sort": {"last_message": -1}
                },
                {
                    "$limit": limit
                }
            ]
            
            sessions = await self.collection.aggregate(pipeline).to_list(length=None)
            
            logger.debug(f"Retrieved {len(sessions)} sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete all messages for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deletion successful, False otherwise
        """
        if not self._connected:
            logger.warning("Not connected to MongoDB, cannot delete session")
            return False
        
        try:
            result = await self.collection.delete_many({"session_id": session_id})
            logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
