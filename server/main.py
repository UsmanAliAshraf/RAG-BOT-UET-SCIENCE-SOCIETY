from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.run_chatbot import run_chatbot
from app.session_manager import session_manager
from typing import Optional

app = FastAPI(title="ECHO - UET Science Society Chatbot", version="1.0.0")

# allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or the frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    memory_content: str
    memory_buffer_length: int

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    """
    Chat endpoint that handles user queries with session management.
    
    If no session_id is provided, a new session will be created.
    If session_id is provided but invalid, a new session will be created.
    """
    try:
        # If no session_id provided, create a new session
        if not query.session_id:
            session_id = session_manager.create_session()
        else:
            # Check if session exists, if not create new one
            session = session_manager.get_session(query.session_id)
            if not session:
                session_id = session_manager.create_session()
            else:
                session_id = query.session_id
        
        # Run the chatbot
        result = run_chatbot(query.question, session_id)
        
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            memory_content=result["memory_content"],
            memory_buffer_length=result["memory_buffer_length"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": session_manager.get_session_count()
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    success = session_manager.delete_session(session_id)
    if success:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/count")
async def get_session_count():
    """Get the current number of active sessions"""
    return {"active_sessions": session_manager.get_session_count()}

@app.get("/session/{session_id}/memory")
async def get_session_memory(session_id: str):
    """Get memory content for a specific session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "memory_content": session.get_memory_content(),
        "memory_buffer_length": len(session.get_memory_buffer())
    }
