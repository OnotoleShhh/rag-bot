import os, glob, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
EMBED_MODEL = os.getenv("EMBED_MODEL")

def read_docs():
    import pypdf
    docs = []
    for path in glob.glob("data/**/*", recursive=True):
        if os.path.isdir(path): 
            continue
        p = Path(path)
        if p.suffix.lower() in [".md", ".txt"]:
            docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
        elif p.suffix.lower() == ".pdf":
            try:
                reader = pypdf.PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                docs.append((p.name, text))
            except Exception:
                pass
    return docs

def chunk(text, max_tokens=350, overlap=50):
    # simple token-ish chunker by characters; good enough to start
    import textwrap
    width = max_tokens * 4
    step = width - overlap*4
    return [text[i:i+width] for i in range(0, len(text), step)]

def main():
    docs = read_docs()
    chunks, meta = [], []
    for fname, txt in docs:
        for ch in chunk(txt):
            if ch.strip():
                chunks.append(ch.strip())
                meta.append({"source": fname})

    model = SentenceTransformer(EMBED_MODEL)
    X = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # FAISS index
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine via normalized dot
    index.add(X)

    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, "storage/index.faiss")
    np.save("storage/meta.npy", np.array(meta, dtype=object))
    with open("storage/chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks: f.write(c.replace("\n", " ") + "\n")

    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs.")

if __name__ == "__main__":
    main()
