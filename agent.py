# agent.py - FAISS + sentence-transformers + pypdf + OpenAI (no LangChain)
import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import math
import time

# utils
from utils.citation import format_sources_from_docs

# Configuration
VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data"
INDEX_FILE = Path(VECTORSTORE_DIR) / "faiss_index.bin"
META_FILE = Path(VECTORSTORE_DIR) / "metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good for semantic search
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# Initialize embedding model (will download on first run)
_EMBED_MODEL = None


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


def pdf_to_pages(path: str) -> List[Dict]:
    """Return list of {'text':..., 'page':i} for each page in PDF."""
    r = PdfReader(path)
    pages = []
    for i, page in enumerate(r.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"text": text.strip(), "page": i})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping chunks of approx chunk_size chars."""
    if not text:
        return []
    text = text.replace("\n", " ").strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def load_pdfs(paths: List[str]) -> List[Dict]:
    """Return list of document chunks with metadata: {'text','source','page'}"""
    docs = []
    for p in paths:
        pages = pdf_to_pages(p)
        fname = Path(p).name
        for page in pages:
            page_text = page.get("text", "")
            page_no = page.get("page", None)
            chunks = chunk_text(page_text)
            for c in chunks:
                docs.append({"text": c, "source": fname, "page": page_no})
    return docs


def build_faiss(docs: List[Dict], persist_directory: str = VECTORSTORE_DIR):
    """Create FAISS index from docs and persist metadata."""
    if not docs:
        raise ValueError("No docs provided to build FAISS.")
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    model = get_embed_model()
    texts = [d["text"] for d in docs]
    # embed in batches
    EMB = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    EMB = np.array(EMB).astype("float32")

    dim = EMB.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(EMB)
    faiss.write_index(index, str(Path(persist_directory) / "faiss_index.bin"))

    # save metadata aligned with embeddings
    with open(Path(persist_directory) / "metadata.pkl", "wb") as f:
        pickle.dump(docs, f)

    return index


def load_faiss(persist_directory: str = VECTORSTORE_DIR):
    idx_path = Path(persist_directory) / "faiss_index.bin"
    meta_path = Path(persist_directory) / "metadata.pkl"
    if not idx_path.exists() or not meta_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs


def build_vectorstore_if_needed(data_dir: str = DATA_DIR, persist_directory: str = VECTORSTORE_DIR):
    """Build FAISS index from PDFs in data/ unless a persisted index already exists."""
    idx, docs = load_faiss(persist_directory)
    if idx is not None and docs is not None:
        return idx, docs

    data_path = Path(data_dir)
    pdfs = [str(p) for p in data_path.glob("*.pdf")]
    if not pdfs:
        raise ValueError("No PDFs found in data/ to index.")
    docs = load_pdfs(pdfs)
    idx = build_faiss(docs, persist_directory)
    return idx, docs


def add_documents_from_paths(paths: List[str], persist_directory: str = VECTORSTORE_DIR):
    """Add (rebuild) vectorstore using the provided PDFs (this implementation rebuilds)."""
    docs = load_pdfs(paths)
    return build_faiss(docs, persist_directory)


def query_index(index, docs_meta: List[Dict], query: str, top_k: int = TOP_K):
    """Return top_k docs for query using local embeddings."""
    model = get_embed_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs_meta):
            continue
        results.append(docs_meta[idx])
    return results


def openai_chat_completion(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 512) -> str:
    """Call OpenAI ChatCompletion (uses openai package)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    openai.api_key = key

    # Chat-style request
    messages = [
        {"role": "system", "content": "You are a helpful HR assistant. Answer concisely and cite sources."},
        {"role": "user", "content": prompt},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.0)
    text = resp["choices"][0]["message"]["content"].strip()
    return text


def ask_hr_assistant(query: str) -> Tuple[str, List[Dict], List[str]]:
    """
    Query the vectorstore and produce an answer + sources list.
    Returns (answer, sources_list) where sources_list are dicts {source,page,snippet}.
    """
    index, docs = load_faiss(VECTORSTORE_DIR)
    if index is None or docs is None:
        index, docs = build_vectorstore_if_needed(DATA_DIR, VECTORSTORE_DIR)

    top_docs = query_index(index, docs, query, top_k=TOP_K)

    # Prepare context snippets
    context_parts = []
    sources_info = []
    for d in top_docs:
        snippet = d.get("text", "")[:1000]
        src = d.get("source", "unknown")
        page = d.get("page", None)
        sources_info.append({"source": src, "page": page, "snippet": snippet})
        context_parts.append(f"Source: {src} {f'| page {page}' if page else ''}\n{snippet}\n---\n")

    context = "\n".join(context_parts)
    prompt = (
        "You are an HR assistant. Use ONLY the context below to answer the question. If not present, say you don't know and suggest contacting HR.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\n\nInstructions:\n- Answer in 2-6 concise sentences.\n- At the end, list sources like: [1] filename (page X)\n"
    )

    answer = openai_chat_completion(prompt)

    # Format sources using helper (returns list like "filename (page X)")
    try:
        formatted = format_sources_from_docs(sources_info)
    except Exception:
        formatted = []

    return answer, sources_info, formatted
