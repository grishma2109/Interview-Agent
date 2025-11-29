import os
import streamlit as st
from pathlib import Path
from agent import (
    build_vectorstore_if_needed,
    add_documents_from_paths,
    ask_hr_assistant,
    VECTORSTORE_DIR,
)

st.set_page_config(page_title="HR Assistant Agent", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

st.title("🧑‍💼 HR Assistant Agent")
st.write("Upload HR policy PDFs, build the index, then ask questions grounded in your documents.")

# Sidebar: Upload PDFs
st.sidebar.header("Upload HR Documents (PDF)")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.write(f"Saving {len(uploaded_files)} file(s) to `data/` ...")
    for f in uploaded_files:
        dest = DATA_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.read())
    st.sidebar.success("Saved uploaded file(s). Click 'Build / Update Index'.")

st.sidebar.markdown("---")
st.sidebar.header("Index / Vectorstore")

if st.sidebar.button("Build / Update Index"):
    with st.spinner("Building vectorstore (this may take a minute)..."):
        pdf_paths = [str(p) for p in DATA_DIR.glob("*.pdf")]
        if not pdf_paths:
            st.sidebar.error("No PDFs found in `data/`. Upload at least one PDF.")
        else:
            add_documents_from_paths(pdf_paths)
            build_vectorstore_if_needed()
            st.sidebar.success("Vectorstore built/updated.")

st.sidebar.markdown("---")
st.sidebar.header("OpenAI Key")

# allow user to paste key for local testing
if ("OPENAI_API_KEY" not in os.environ) and (not hasattr(st, "secrets") or "OPENAI_API_KEY" not in st.secrets):
    key = st.sidebar.text_input(
        "Enter OpenAI API key (or set as env var OPENAI_API_KEY)",
        type="password"
    )
    if key:
        os.environ["OPENAI_API_KEY"] = key
        st.sidebar.success("API key set for this session.")
else:
    st.sidebar.write("OpenAI key found.")

st.markdown("## Ask the HR Assistant")
query = st.text_input("Enter your question here", key="query_input")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Ask"):
        if not query or query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources, formatted = ask_hr_assistant(query)
                except Exception as e:
                    st.error(f"Error: {e}")
                    answer = None
                    sources = None
                    formatted = None

            if answer:
                st.markdown("### Answer")
                st.success(answer)

                if (sources and len(sources) > 0) or (formatted and len(formatted) > 0):
                    st.markdown("### Sources / Evidence")
                    # Prefer the formatted strings for the title, but keep snippets
                    if formatted:
                        for i, fsrc in enumerate(formatted, 1):
                            st.markdown(f"**{i}.** {fsrc}")
                            if sources and i-1 < len(sources):
                                snippet = sources[i-1].get("snippet")
                                if snippet:
                                    st.write(f"> {snippet}")
                    else:
                        for i, s in enumerate(sources, 1):
                            src = s.get("source", "unknown")
                            page = s.get("page", None)
                            snippet = s.get("snippet", None)
                            st.markdown(
                                f"**{i}.** `{src}`" + (f" — page {page}" if page is not None else "")
                            )
                            if snippet:
                                st.write(f"> {snippet}")

st.markdown("---")
st.info(
    "This assistant answers using uploaded documents only. For live HR actions, integrate with your internal HR systems."
)
