import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
VS_DIR = "vectorstore"

def load_docs():
    docs = []
    # PDF
    for path in [p for p in os.listdir(DATA_DIR) if p.lower().endswith(".pdf")]:
        docs.extend(PyPDFLoader(os.path.join(DATA_DIR, path)).load())
    # TXT/MD
    for path in [p for p in os.listdir(DATA_DIR) if p.lower().endswith((".txt",".md"))]:
        docs.extend(TextLoader(os.path.join(DATA_DIR, path), autodetect_encoding=True).load())
    return docs

def main():
    docs = load_docs()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, emb)
    os.makedirs(VS_DIR, exist_ok=True)
    vs.save_local(VS_DIR)
    print(f"Indexed {len(docs)} chunks into {VS_DIR}/")

if __name__ == "__main__":
    main()
