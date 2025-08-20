# LangGraph-style chatbot (single-file)
# Requirements: langchain, langchain_community, langchain_groq, dotenv, faiss, sentence-transformers
# Designed to replace your previous LangChain file; uses in-memory session-only memory_graph and FAISS retriever.

import os
import re
import warnings
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_groq.chat_models import ChatGroq

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------
# 0️⃣ Load env + keys
# ------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set environment variable GROQ_API_KEY.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ------------------------------
# 1️⃣ Globals / State Initialization
# ------------------------------
state = {
    "question": None,        # user input
    "answer": None,          # final output
    "memory_graph": [],      # list of triplets e.g. [("user","name","Usman")]
    "debug_trace": [],       # visited nodes in order
}

# Hardcoded keyword list (20 terms) for rule-based routing
KEYWORDS = [
    "hello", "hi", "hey", "bye", "goodbye",
    "thanks", "thank", "ok", "okay",
    "time", "date", "contact", "info", "help", "i am"
]


# Regex patterns for simple fact extraction (expandable)
FACT_PATTERNS = [
    # ("subject", "predicate", regex_to_extract_object)
    ("user", "name", r"\b(?:my name is|I'm|I am|name is)\s+([A-Z][a-zA-Z]{1,30})\b"),
    ("user", "location", r"\b(?:I live in|I'm from|I am from)\s+([A-Za-z ,]{2,50})\b"),
    ("user", "profession", r"\b(?:I am a|I'm a|I work as a)\s+([A-Za-z ]{2,40})\b"),
]

# ------------------------------
# 2️⃣ Load Embeddings + FAISS (you said index already exists locally)
# ------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Make sure the directory "faiss_index" is the one you already created
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Retriever with MMR (match your previous settings)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
    streaming=False,
    verbose=False,
)

# ------------------------------
# 3️⃣ Define the LLM (ChatGroq)
# ------------------------------
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.0,
    max_tokens=512,
)

# We'll use LLMChain to call the LLM in an explicit Answer Node
# ------------------------------
# 4️⃣ Prompts
# ------------------------------
# Retriever / general prompt - we always pass through LLM to avoid logic divergence
BASE_PROMPT_TEMPLATE = """
You are a professional assistant for UET Science Society. Use the CONTEXT and MEMORY to answer concisely.
If the answer is not present in the context or memory, say "I don't know." and do NOT hallucinate.
Do not add extra information or make up sources.

CONTEXT:
{context}

MEMORY:
{memory}

QUESTION:
{question}
"""

base_prompt = PromptTemplate(
    input_variables=["context", "memory", "question"],
    template=BASE_PROMPT_TEMPLATE
)

# LLMChain for the Answer Node
answer_chain = LLMChain(llm=llm, prompt=base_prompt)

# ------------------------------
# Helper utilities (nodes implemented as funcs)
# ------------------------------
def input_node(user_text: str, st: Dict[str, Any]):
    st["question"] = user_text.strip()
    st["debug_trace"].append("InputNode")
    return st

def router_node(st: Dict[str, Any]) -> str:
    """
    Rule-based routing:
      - If question contains any keyword => 'general_answer'
      - Else if memory_graph has a match => 'memory_answer'
      - Else => 'retriever'
    """
    st["debug_trace"].append("RouterNode")
    q = st["question"].lower()
    # keyword rule
    for kw in KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q):
            return "general_answer"
    # memory match rule: check if any memory value word appears in question
    for (s, p, o) in st["memory_graph"]:
        if o and re.search(r"\b" + re.escape(str(o).lower()) + r"\b", q):
            return "memory_answer"
    # default retriever
    return "retriever"

def general_answer_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a short context from nothing (or small canned help) and call LLM for safe, concise answer.
    """
    st["debug_trace"].append("GeneralAnswerNode")
    # small canned context to avoid hallucination on trivial keywords
    canned_context = "This is an assistant for UET Science Society to help users with queries related to UET Science Society."
    memory_text = format_memory(st["memory_graph"])
    prompt_inputs = {
        "context": canned_context,
        "memory": memory_text,
        "question": st["question"]
    }
    out = answer_chain.run(prompt_inputs)
    st["answer"] = out.strip()
    return st

def memory_answer_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use memory facts as context and call LLM for answer. Keeps LLM usage consistent.
    """
    st["debug_trace"].append("MemoryAnswerNode")
    memory_text = format_memory(st["memory_graph"])
    prompt_inputs = {"context": "Use memory facts below.", "memory": memory_text, "question": st["question"]}
    out = answer_chain.run(prompt_inputs)
    st["answer"] = out.strip()
    return st

def retriever_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch top docs from FAISS retriever, assemble context, and call LLM.
    """
    st["debug_trace"].append("RetrieverNode")
    query = st["question"]
    docs = retriever.get_relevant_documents(query)  # returns a list of Document objects
    # assemble context: join top doc pages/texts but keep short
    context_snippets = []
    for d in docs[:3]:
        txt = getattr(d, "page_content", str(d))
        # truncate to first 800 chars per doc to avoid token bloat
        context_snippets.append(txt[:800])
    context = "\n---\n".join(context_snippets) if context_snippets else "No context found."
    memory_text = format_memory(st["memory_graph"])
    prompt_inputs = {"context": context, "memory": memory_text, "question": st["question"]}
    out = answer_chain.run(prompt_inputs)
    st["answer"] = out.strip()
    return st

def fallback_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe default if nothing matched.
    """
    st["debug_trace"].append("FallbackNode")
    st["answer"] = "I don't know."
    return st

def answer_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    The "Answer Node" here is effectively covered by the LLM calls in above nodes.
    We keep this function to conform to your requested graph: it just logs entry.
    """
    st["debug_trace"].append("AnswerNode")
    # already set in the respective branch; no-op
    return st

def memory_update_node(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract simple facts using regex and append to memory_graph as triplets.
    Avoid duplicates.
    """
    st["debug_trace"].append("MemoryUpdateNode")
    text = st["question"]
    for subj, pred, pattern in FACT_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            val = m.strip()
            triplet = (subj, pred, val)
            if triplet not in st["memory_graph"]:
                st["memory_graph"].append(triplet)
    return st

def format_memory(mem: List[Tuple[str, str, str]]) -> str:
    if not mem:
        return "No memory."
    return "\n".join([f"{s}.{p} = {o}" for (s, p, o) in mem])

# ------------------------------
# 5️⃣ Main loop (CLI)
# ------------------------------
def run_chat():
    print("🤖 LangGraph-style ChatBot (ChatGroq + FAISS) — type 'exit' to quit\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        # reset debug trace for this turn
        state["debug_trace"] = []

        # 1. Input Node
        input_node(user_input, state)

        # 2. Router Node
        route = router_node(state)

        # 3. Branching (General / Memory / Retriever). All paths call the LLM to avoid hallucination risk.
        if route == "general_answer":
            general_answer_node(state)
        elif route == "memory_answer":
            memory_answer_node(state)
        elif route == "retriever":
            retriever_node(state)
        else:
            fallback_node(state)

        # 4. Answer Node (logs)
        answer_node(state)

        # 5. Memory Update Node (extract facts and update session memory)
        memory_update_node(state)

        # 6. Debug prints (as requested)
        print("\nBot:", state["answer"], "\n")
        print("----- DEBUG TRACE -----")
        # show visited nodes
        print("Visited nodes:", " -> ".join(state["debug_trace"]))
        # show memory snapshot
        print("Memory graph snapshot:", state["memory_graph"])
        print("-----------------------\n")

# Entry point
if __name__ == "__main__":
    run_chat()
