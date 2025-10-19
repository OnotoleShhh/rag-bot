import os, faiss, numpy as np, requests
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL", "llama3.1:8b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# load index + metadata
index = faiss.read_index("storage/index.faiss")
meta  = np.load("storage/meta.npy", allow_pickle=True)
with open("storage/chunks.txt", encoding="utf-8") as f:
    chunks = [line.strip() for line in f]

def retrieve(query, k=5):
    from sentence_transformers import SentenceTransformer
    em = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(em)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q, k)
    ctx = [(chunks[i], meta[i]["source"], scores[0][j]) for j,i in enumerate(idx[0])]
    return ctx

def build_prompt(query, contexts):
    context_block = "\n\n".join(
        [f"[{j+1}] (source: {src})\n{c}" for j,(c,src,_) in enumerate(contexts)]
    )
    return (
        "Answer using only the context if possible. If it’s not in the context, say you don’t know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )

def answer(query):
    ctx = retrieve(query, k=5)
    prompt = build_prompt(query, ctx)

    payload = {
        "model": MODEL,
        "messages": [
            {"role":"system","content":"You are a helpful assistant for RAG."},
            {"role":"user","content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    return content, ctx

if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk: ").strip()
            if not q: continue
            ans, ctx = answer(q)
            print("\n=== Answer ===")
            print(ans)
            print("\n=== Sources ===")
            for j,(c,src,score) in enumerate(ctx,1):
                print(f"{j}. {src} (score={score:.3f})")
        except (EOFError, KeyboardInterrupt):
            break
