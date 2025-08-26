import re
from app.app import run_chat_pipeline  # import the new pipeline function

def clean_think_tags(text: str) -> str:
    """
    Comprehensive cleaning function to remove all variations of think tags
    and any remaining internal reasoning text.
    """
    if not text:
        return text
    
    # Remove <think>...</think> blocks (case insensitive, multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove <thinking>...</thinking> blocks
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove [think]...[/think] blocks
    cleaned = re.sub(r"\[think\].*?\[/think\]", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining think-related patterns
    cleaned = re.sub(r"<think.*?>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any lines that start with "think:" or "thinking:"
    cleaned = re.sub(r"^(think|thinking):.*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up extra whitespace and empty lines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove multiple empty lines
    cleaned = cleaned.strip()
    
    return cleaned

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
        
        # Clean the answer thoroughly
        raw_answer = result["answer"]
        clean_answer = clean_think_tags(raw_answer)
        
        # Also clean memory content to be safe
        clean_memory = clean_think_tags(result["memory_content"])
        
        # Return the result with cleaned answer and memory
        return {
            "answer": clean_answer,
            "memory_content": clean_memory,
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
