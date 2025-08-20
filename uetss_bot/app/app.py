from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os
import warnings
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------------------
# 2️⃣ Load embeddings + FAISS index
# ------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# ------------------------------
# 3️⃣ Create retriever with MMR
# ------------------------------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
    streaming=False,
    verbose=False
)

# ------------------------------
# 4️⃣ Define LLM (Groq)
# ------------------------------
os.environ["GROQ_API_KEY"] = groq_api_key
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=512
)

# ------------------------------
# 5️⃣ Define prompt
# ------------------------------
template = """
You are a professional assistant for UET Science Society.
Use the context below to answer the question concisely. 
If the answer is not present, say "I don't know, Please ask only about science society.".
Do not provide any explanations or additional information.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

# ------------------------------
# 6️⃣ Add memory (last 3 turns only)
# ------------------------------
memory = ConversationBufferWindowMemory(
    k=3,                       # last 3 exchanges
    memory_key="chat_history", # key name must match prompt input
    return_messages=True
)

# ------------------------------
# 7️⃣ Conversational Retrieval Chain (QA + Memory)
# ------------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
)

# ------------------------------
# Chat CLI loop
# ------------------------------
print("🤖 UET Science Society ChatBot (type 'exit' to quit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        print("Goodbye! 👋")
        break

    result = qa_chain.invoke({"question": query})
    raw_answer = result['answer']

    print(f"Bot: {raw_answer}\n")
