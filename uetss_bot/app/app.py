from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os
import warnings
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
# 4️⃣ Define LLM (Groq) in LangChain
# ------------------------------
# Make sure you set your Groq API key in env: GROQ_API_KEY
os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(
    model="qwen/qwen3-32b",   # Groq model suitable for QA / RAG
    temperature=0,
    max_tokens=512
)

# ------------------------------
# 5️⃣ Create a prompt template
# ------------------------------
template = """
You are a professional assistant for UET Science Society.
Use the context below to answer the question concisely. If the answer is not present, say "I don't know, Please ask only about science society.".
Do not provide any explanations or additional information.

CONTEXT:
{context}

QUESTION:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ------------------------------
# 6️⃣ Create RetrievalQA chain
# ------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",        # Stuff means combine all chunks into prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ------------------------------
# Chat CLI loop (clean, no verbose)
# ------------------------------
print("🤖 UET Science Society ChatBot (type 'exit' to quit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Ask the QA chain, disable verbose
    result = qa_chain.invoke({"query": query})  # use .invoke() instead of .__call__ to avoid warnings
    raw_answer = result['result']

    # Remove everything between <think> and </think>
    import re
    clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()

    print(f"Bot: {clean_answer}\n")

