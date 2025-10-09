import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VS_DIR = "vectorstore"
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(VS_DIR, emb, allow_dangerous_deserialization=True)

app = FastAPI(title="RAG Bot (baseline)")

class Query(BaseModel):
    question: str
    k: int = 4

@app.post("/ask")
def ask(q: Query):
    docs = db.similarity_search(q.question, k=q.k)
    # На шаге 1: просто возвращаем найденные куски как «контекст».
    # На шаге 2 подключим LLM и сделаем генерацию ответа.
    return {
        "question": q.question,
        "contexts": [d.page_content[:800] for d in docs]
    }
