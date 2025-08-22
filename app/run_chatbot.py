import re
from app.app import state, input_node, retriever_node, answer_node  # import from app/app.py

def run_chatbot(query: str) -> str:
    """
    Runs the chatbot for a given query and returns only the final clean answer
    (removes <think>...</think> blocks).
    """
    # Reset debug trace
    state["debug_trace"] = []

    # Run pipeline
    input_node(query, state)
    retriever_node(state)
    answer_node(state)

    raw_answer = state["answer"]

    # Remove <think>...</think> section
    clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    return clean_answer
