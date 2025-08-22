# LangGraph-style chatbot (single-file)
# Requirements: langchain, langchain_community, langchain_groq, dotenv, faiss, sentence-transformers

import os
import warnings
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_groq.chat_models import ChatGroq
from langchain.memory import ConversationSummaryMemory

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
    "debug_trace": [],       # visited nodes in order
}

# ------------------------------
# 2️⃣ Load Embeddings + FAISS
# ------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

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

# ------------------------------
# 4️⃣ Memory (Summary-based)
# ------------------------------
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="memory",   # will inject into prompt
    input_key="question"   # aligns with our state["question"]
)

# ------------------------------
# 5️⃣ Prompts
# ------------------------------
BASE_PROMPT_TEMPLATE = """
Your name is vivi.
You are a professional yet friendly assistant for UET Science Society, here to help users with their queries about the society. 
Use the CONTEXT and MEMORY to answer concisely.
If the user greets or ask about you greet them back and if they ask about you, tell them about yourself.
If the answer is not present in the context or memory, say in a nice apologetic way that you don't know.
Do NOT hallucinate, keep things simple.
Do not add extra information or make up sources.
Do not tell them anything about context or data sources, i.e "this thing is listed twice in context or memory".
and Again, keep your answers concise.

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

answer_chain = LLMChain(llm=llm, prompt=base_prompt, memory=memory)

# ------------------------------
# Helper utilities (nodes implemented as funcs)
# ------------------------------
def input_node(user_text: str, st: Dict[str, Any]):
    st["question"] = user_text.strip()
    st["debug_trace"].append("InputNode")
    return st

def retriever_node(st: Dict[str, Any]) -> Dict[str, Any]:
    st["debug_trace"].append("RetrieverNode")
    query = st["question"]
    docs = retriever.get_relevant_documents(query)
    context_snippets = []
    for d in docs[:3]:
        txt = getattr(d, "page_content", str(d))
        context_snippets.append(txt[:800])
    context = "\n---\n".join(context_snippets) if context_snippets else "No context found."

    # The memory object will inject past summary automatically
    prompt_inputs = {"context": context, "question": st["question"]}
    out = answer_chain.run(prompt_inputs)
    st["answer"] = out.strip()
    return st

def answer_node(st: Dict[str, Any]) -> Dict[str, Any]:
    st["debug_trace"].append("AnswerNode")
    return st

# ------------------------------
# 6️⃣ Main loop (CLI)
# ------------------------------
def run_chat():
    print("🤖 LangGraph-style ChatBot (ChatGroq + FAISS + SummaryMemory) — type 'exit' to quit\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        state["debug_trace"] = []
        input_node(user_input, state)

        retriever_node(state)
        answer_node(state)

        print("\nBot:", state["answer"], "\n")
        print("----- DEBUG TRACE -----")
        print("Visited nodes:", " -> ".join(state["debug_trace"]))
        print("-----------------------\n")

if __name__ == "__main__":
    run_chat()
