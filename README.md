# 🤖 ECHO – UET Science Society Chatbot

An AI-powered **chatbot** for the **UET Science Society** that uses a **Retrieval-Augmented Generation (RAG)** pipeline built in LangGraph.  
It can answer FAQs, provide event information, and serve as a smart assistant for students and community members.  

Built with:
- ⚡ **FastAPI** – lightweight backend API
- 🧠 **Custom RAG pipeline** – keyword + retriever + answer generator
- 🌐 **Vanilla JS + HTML** – simple frontend
- 🔄 **CORS enabled** – easy local testing with frontend

---

## 🚀 Features
- Ask natural language questions through a **chat interface**.
- Backend pipeline with:
  - `input_node` → handles user input.
  - `retriever_node` → retrieves possible answers from knowledge base.
  - `answer_node` → generates the final response.
- Removes unwanted `<think>...</think>` reasoning text before sending response to user.
- Fully containerizable and easy to deploy.

---

## 📂 Project Structure
```

RAG-BOT-UET-SCIENCE-SOCIETY/
│
├── app/
│   ├── app.py           # Core chatbot logic (state, nodes, pipeline)
│   ├── run\_chatbot.py   # Chatbot runner (cleaned output)
|   |___ prepare\_data.py # this creats faiss indexes from raw data
│   └── **init**.py      # Empty (or exports for easier imports)
│
├── server/
│   └── main.py          # FastAPI server (API endpoint for chatbot)
│
├── static/
│   └── index.html       # Frontend UI (simple chat interface)
│
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies

````

---

## 🛠️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/RAG-BOT-UET-SCIENCE-SOCIETY.git
cd RAG-BOT-UET-SCIENCE-SOCIETY
````

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI server

From the **project root**:

```bash
uvicorn server.main:app --reload
```

API will be available at:
👉 `http://127.0.0.1:8000`

---

## 💻 Frontend (Local)

1. Open `static/index.html` in your browser.
2. Type a question in the input box.
3. The chatbot will respond using the FastAPI backend.

---

## 🧹 Notes

* The backend removes internal reasoning (`<think>...</think>`) before sending answers to the frontend.
* Knowledge base for retriever can be updated inside `app/app.py`.
* You can extend the frontend with a proper chat UI or integrate into your official website.

---

## 📜 License
All Rights of this chatbot belongs to UET Science Society - © 2025.

---

## 👥 Contributors

* **Usman Ali Ashraf** – Lead Developer
---

