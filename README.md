# HR Assistant Agent

**A Retrieval-Augmented Generation (RAG) HR assistant** that answers employee queries about policies, leave rules, benefits and onboarding by reading uploaded documents (PDFs).

## Features
- Upload HR documents (PDFs) and build a vector index.
- Ask natural language questions; answers are grounded in uploaded documents.
- Shows short evidence snippets and source file names (page numbers where available).
- Simple Streamlit UI for quick demo and deployment.

## Tech stack
- UI: Streamlit
- RAG: LangChain (document loaders, text splitter)
- Embeddings: OpenAIEmbeddings (OpenAI API key required)
- Vector DB: Chroma (persisted to `vectorstore/`)
- LLM: OpenAI (for final answer synthesis)

## Setup (local)
1. Clone repo
```bash
git clone <your-repo-url>
cd hr-assistant
