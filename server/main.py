from fastapi import FastAPI
from pydantic import BaseModel
from app.run_chatbot import run_chatbot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer = run_chatbot(query.question)
    return {"answer": answer}
