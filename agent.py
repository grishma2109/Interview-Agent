# agent.py - FAISS + OpenAI embeddings + pypdf (no sentence-transformers / no TF)
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import faiss
import openai
from pypdf import PdfReader

# utils (keeps your existing citation helper)
from utils.citation import format_sources_from_docs

# Configuration
VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data"
INDEX_FILE = Path(VECTORSTORE_DIR) / "faiss_index.bin"
META_FILE = Path(VECTORSTORE_DIR) / "metadata.pkl"

# Embedding + chunking params
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# Ensure OPENAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    # don't raise here; functions will check and raise with clearer message
    pass
else:
    openai.api_key = OPENAI_API_KEY


# -------- PDF loading & chunking --------
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
    """
    Load PDFs and return list of document chunks with metadata:
      {'text': chunk_text, 'source': filename, 'page': page_no}
    """
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


# -------- OpenAI Embeddings wrapper --------
def _check_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set environment variable or Streamlit secret before building/asking."
        )
    openai.api_key = key


def embed_texts_openai(texts: List[str], model: str = EMBED_MODEL, batch_size: int = 128) -> np.ndarray:
    """
    Get embeddings from OpenAI Embeddings API.
    Returns numpy array shape (N, D) dtype float32.
    """
    _check_openai_key()
    all_embs: List[List[float]] = []
    # chunk into batches to avoid huge requests
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # OpenAI SDK: create embeddings
        resp = openai.Embedding.create(model=model, input=batch)
        embs = [r["embedding"] for r in resp["data"]]
        all_embs.extend(embs)
    arr = np.array(all_embs, dtype="float32")
    return arr


# -------- FAISS index helpers --------
def build_faiss_from_docs(docs: List[Dict], persist_dir: str = VECTORSTORE_DIR) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from docs (list of {'text','source','page'}).
    Persists index and metadata to VECTORSTORE_DIR.
    Returns the FAISS index object.
    """
    if not docs:
        raise ValueError("No docs provided to build FAISS index.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    texts = [d["text"] for d in docs]
    emb = embed_texts_openai(texts)  # (N, D) float32
    dim = emb.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    faiss.write_index(index, str(Path(persist_dir) / "faiss_index.bin"))

    # Save metadata aligned with embeddings
    with open(Path(persist_dir) / "metadata.pkl", "wb") as f:
        pickle.dump(docs, f)

    return index


def load_faiss(persist_dir: str = VECTORSTORE_DIR):
    """
    Load persisted FAISS index and metadata if present.
    Returns (index, docs) or (None, None) if not found.
    """
    idx_path = Path(persist_dir) / "faiss_index.bin"
    meta_path = Path(persist_dir) / "metadata.pkl"
    if not idx_path.exists() or not meta_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs


# -------- Public API (compatible signatures) --------
def build_vectorstore_if_needed(data_dir: str = DATA_DIR, persist_directory: str = VECTORSTORE_DIR):
    """
    Build FAISS vectorstore from PDFs in data/ unless persisted index exists.
    Returns (index, docs).
    """
    idx, docs = load_faiss(persist_directory)
    if idx is not None and docs is not None:
        return idx, docs

    data_path = Path(data_dir)
    pdfs = [str(p) for p in data_path.glob("*.pdf")]
    if not pdfs:
        raise ValueError("No PDFs found in data/ to index.")
    docs = load_pdfs(pdfs)
    idx = build_faiss_from_docs(docs, persist_directory)
    return idx, docs


def add_documents_from_paths(paths: List[str], persist_directory: str = VECTORSTORE_DIR):
    """
    Load PDFs at `paths`, chunk and (re)build FAISS vectorstore.
    Returns the FAISS index.
    """
    docs = load_pdfs(paths)
    idx = build_faiss_from_docs(docs, persist_directory)
    return idx


def query_index(index: faiss.IndexFlatL2, docs_meta: List[Dict], query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Query the FAISS index with OpenAI embeddings for `query`.
    Returns list of top_k metadata dicts aligned with docs_meta.
    """
    if index is None or docs_meta is None:
        raise ValueError("Index or docs_meta is None.")

    q_emb = embed_texts_openai([query])  # (1, D)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs_meta):
            continue
        results.append(docs_meta[idx])
    return results


# -------- OpenAI Chat completion wrapper --------
def openai_chat_completion(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 512) -> str:
    """
    Use OpenAI ChatCompletion to generate an answer from the prompt.
    """
    _check_openai_key()
    messages = [
        {"role": "system", "content": "You are a helpful HR assistant. Answer concisely and cite sources."},
        {"role": "user", "content": prompt},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.0)
    text = resp["choices"][0]["message"]["content"].strip()
    return text


def ask_hr_assistant(query: str) -> Tuple[str, List[Dict], List[str]]:
    """
    Query the local FAISS vectorstore and synthesize an answer using OpenAI chat.
    Returns (answer_text, sources_info, formatted_sources)
      - sources_info: list of dicts {source, page, snippet}
      - formatted_sources: result of utils.citation.format_sources_from_docs (if available)
    """
    # Ensure OPENAI_API_KEY
    _check_openai_key()

    # Ensure index exists
    idx, docs = load_faiss(VECTORSTORE_DIR)
    if idx is None or docs is None:
        idx, docs = build_vectorstore_if_needed(DATA_DIR, VECTORSTORE_DIR)

    top_docs = query_index(idx, docs, query, top_k=TOP_K)

    # Prepare context and sources
    sources_info = []
    context_parts = []
    for d in top_docs:
        snippet = d.get("text", "")[:1000]
        src = d.get("source", "unknown")
        page = d.get("page", None)
        sources_info.append({"source": src, "page": page, "snippet": snippet})
        context_parts.append(f"Source: {src} {f'| page {page}' if page else ''}\n{snippet}\n---\n")

    context = "\n".join(context_parts) if context_parts else "No context available."

    prompt = f"""
You are an HR assistant. Use ONLY the information from the context below to answer the question.
If the answer is not contained in the context, say you don't know and suggest contacting HR.

Context:
{context}

Question:
{query}

Instructions:
- Provide a concise answer (2-6 sentences).
- At the end, list the sources you used in the form: [1] filename (page X)
- If multiple relevant sources exist, synthesize them.
"""

    answer_text = openai_chat_completion(prompt)

    # Attempt to format sources (silently continue on failure)
    try:
        formatted = format_sources_from_docs(sources_info)
    except Exception:
        formatted = []

    return answer_text, sources_info, formatted
