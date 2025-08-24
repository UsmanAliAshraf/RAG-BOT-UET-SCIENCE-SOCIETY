import uuid
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

@dataclass
class ChatSession:
    """Represents a single chat session with its state and metadata"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Session state
    question: Optional[str] = None
    answer: Optional[str] = None
    debug_trace: list = field(default_factory=list)
    
    # Memory for this session - will be set when first created
    memory: Optional[Any] = None  # ConversationSummaryMemory instance
    
    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = datetime.now()
    
    def reset_state(self):
        """Reset the current conversation state (but keep memory)"""
        self.question = None
        self.answer = None
        self.debug_trace = []
    
    def get_memory_content(self) -> str:
        """Get the current memory content as a string"""
        if self.memory:
            try:
                # Get the memory buffer content - use the correct attribute
                memory_content = self.memory.buffer
                return memory_content if memory_content else "No memory yet"
            except Exception as e:
                return f"Error accessing memory: {str(e)}"
        return "No memory initialized"
    
    def get_memory_buffer(self) -> list:
        """Get the raw memory buffer"""
        if self.memory:
            try:
                return self.memory.chat_memory.messages
            except Exception as e:
                return []
        return []

class SessionManager:
    """Manages chat sessions with automatic cleanup"""
    
    def __init__(self, session_timeout_minutes: int = 10):
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def create_session(self) -> str:
        """Create a new session and return its ID"""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id=session_id)
        
        with self.lock:
            self.sessions[session_id] = session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID, updating its activity timestamp"""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self.lock:
            return self.sessions.pop(session_id, None) is not None
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have exceeded the timeout"""
        now = datetime.now()
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                if now - session.last_activity > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
        
        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Run every 1 minutes
                self.cleanup_expired_sessions()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def get_session_count(self) -> int:
        """Get the current number of active sessions"""
        with self.lock:
            return len(self.sessions)

# Global session manager instance
session_manager = SessionManager()
