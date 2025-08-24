import re
from app.app import run_chat_pipeline  # import the new pipeline function

def run_chatbot(query: str, session_id: str = None) -> dict:
    """
    Runs the chatbot for a given query and returns the answer and memory info
    (removes <think>...</think> blocks).
    
    Args:
        query: The user's question
        session_id: Optional session ID. If None, a new session will be created
    
    Returns:
        Dictionary with answer, memory_content, memory_buffer_length, and session_id
    """
    try:
        # Run the pipeline with session management
        result = run_chat_pipeline(session_id, query)
        
        # Remove <think>...</think> section from answer
        raw_answer = result["answer"]
        clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
        
        # Return the result with cleaned answer
        return {
            "answer": clean_answer,
            "memory_content": result["memory_content"],
            "memory_buffer_length": result["memory_buffer_length"],
            "session_id": result["session_id"]
        }
        
    except Exception as e:
        # Return a user-friendly error message
        return {
            "answer": "I apologize, but I encountered an error processing your request. Please try again.",
            "memory_content": "Error occurred",
            "memory_buffer_length": 0,
            "session_id": session_id or "error"
        }
