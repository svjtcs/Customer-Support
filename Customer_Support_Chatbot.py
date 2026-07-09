"""
==============================================================================
 RAG Customer Support Chatbot - Streamlit UI + swappable LLM (Claude / Gemini)
==============================================================================
 Wraps the RAG pipeline (load -> clean -> chunk -> embed -> FAISS index ->
 retrieve -> augment -> generate) in a Streamlit chat interface. The
 generation backend is SWAPPABLE: pick Claude or Google Gemini from the
 sidebar. Retrieval is identical for both - only the final API call branches.

 PERFORMANCE:
   - The FAISS index + docs are BUILT ONCE and SAVED to disk. On every later
     startup they're loaded from disk (near-instant) instead of re-embedding
     ~thousands of chunks. Delete faiss.index / mini_docs.npy to force a rebuild
     (e.g. after changing the CSV or the chunking logic).
   - Chunking keeps short tickets as a SINGLE chunk and only splits long ones,
     which reduces the chunk count and improves retrieval coherence.

 SETUP:
   pip install streamlit pandas sentence-transformers faiss-cpu anthropic google-generativeai

   Put Customer_Support_Training_Dataset.csv in the same folder as this file.

   Add your keys to .streamlit/secrets.toml:
     ANTHROPIC_API_KEY = "sk-ant-..."
     GEMINI_API_KEY    = "AIza..."

 RUN:
   streamlit run rag_streamlit_app.py
==============================================================================
"""

import os
import re
import streamlit as st
import pandas as pd
import numpy as np                            # <-- NEW: for saving/loading docs
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic
import google.generativeai as genai

st.set_page_config(page_title="Customer Support Bot", page_icon="🎧", layout="centered")

CSV_PATH = "Customer_Support_Training_Dataset.csv"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap; swap to sonnet for harder cases
GEMINI_MODEL = "gemini-2.5-flash"            # free-tier friendly Gemini model

INDEX_FILE = "faiss.index"                    # <-- NEW: saved FAISS index
DOCS_FILE = "mini_docs.npy"                   # <-- NEW: saved chunk metadata
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"         # <-- NEW: named for reuse


# =============================================================================
# STAGE 1-4: Load, clean, chunk, embed, and build the FAISS index.
#
# Fix 1 (disk cache): if a saved index + docs already exist on disk, load them
#   instead of re-embedding everything. This makes every startup after the
#   first near-instant (crucial for hosting, where cold starts are common).
#
# Fix 2 (smarter chunking): short tickets stay as ONE chunk; only long tickets
#   are split. Fewer, more coherent chunks -> faster embedding + better recall.
# =============================================================================
@st.cache_resource(show_spinner="Loading knowledge base and building search index...")
def build_index():
    # Embedder is always needed (for encoding live queries), so load it first.
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # ---- Fix 1: fast path - load prebuilt index from disk if present --------
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(INDEX_FILE)
        mini_docs = np.load(DOCS_FILE, allow_pickle=True).tolist()
        return embedder, index, mini_docs

    # ---- Otherwise, build from scratch (first run only) ---------------------

    # ---- Stage 1: Load ------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # ---- Stage 2: Clean ------------------------------------------------------
    def clean(s):
        s = str(s)
        s = re.sub(r"\{\{.*?\}\}", "", s)      # strip {{placeholder}} tags
        s = re.sub(r"\s+", " ", s).strip()      # collapse whitespace
        return s.lower()

    df["instruction_clean"] = df["instruction"].map(clean)
    df["response_clean"] = df["response"].map(clean)

    # ---- Stage 3: Chunk (Fix 2: one chunk per ticket unless it's long) ------
    def chunk_words(text, n=200, overlap=30):
        """Split into ~n-word chunks with overlap. If the text already fits in
        one chunk (<= n words), return it whole - most support tickets are short
        and shouldn't be fragmented."""
        words = text.split()
        if len(words) <= n:                    # <-- Fix 2: keep short tickets intact
            return [text]
        step = max(1, n - overlap)
        return [" ".join(words[i:i + n]) for i in range(0, len(words), step)]

    mini_docs = []
    for rid, row in df.iterrows():
        body = f"instruction: {row['instruction_clean']} | response: {row['response_clean']}"
        for j, ch in enumerate(chunk_words(body)):
            mini_docs.append({"rid": int(rid), "chunk_id": j, "text": ch})

    # ---- Stage 4: Embed + FAISS index ----------------------------------------
    chunk_texts = [d["text"] for d in mini_docs]
    X = embedder.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=256,                        # <-- batch for a bit more speed
        show_progress_bar=True,                # <-- see progress in the terminal
    ).astype("float32")

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # ---- Fix 1: save to disk so future startups skip the rebuild ------------
    faiss.write_index(index, INDEX_FILE)
    np.save(DOCS_FILE, np.array(mini_docs, dtype=object))

    return embedder, index, mini_docs


# =============================================================================
# STAGE 5: RETRIEVE -> AUGMENT -> GENERATE  (routes to Claude OR Gemini)
# =============================================================================
def retrieve(embedder, index, mini_docs, question, k=3, pool=30):
    """RETRIEVE + start of AUGMENT: embed the question, search FAISS,
    return the top-k matching chunks."""
    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    _, I = index.search(qv, pool)
    hits = [mini_docs[int(idx)] for idx in I[0][:k]]
    return hits


def build_context(hits):
    """AUGMENT: format retrieved chunks into a labeled, citable context block."""
    return "\n\n".join([
        f"[Doc {i+1}] (rid={h['rid']}, chunk={h['chunk_id']})\n{h['text']}"
        for i, h in enumerate(hits)
    ])


SYSTEM_PROMPT = (
    "You are a helpful customer support assistant.\n"
    "Use only the provided context to answer the user's question.\n"
    "If the answer is not available in the context, say: "
    "'Sorry, I don't have that information.'\n"
    "Be polite, concise, and cite sources like [Doc 1], [Doc 2] when relevant."
)


def stream_claude_answer(api_key, context, question, history):
    """GENERATE (Claude): call Claude with the augmented prompt, streaming."""
    client = Anthropic(api_key=api_key)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = history + [{"role": "user", "content": user_prompt}]

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_gemini_answer(api_key, context, question, history):
    """GENERATE (Gemini): same augmented prompt, streaming. Gemini uses
    system_instruction instead of a system field, and role 'model' for the
    assistant."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    gemini_history = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [m["content"]]}
        for m in history
    ]

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_answer(provider, keys, context, question, history):
    """Route generation to the selected LLM backend."""
    if provider == "Claude":
        yield from stream_claude_answer(keys["anthropic"], context, question, history)
    elif provider == "Gemini":
        yield from stream_gemini_answer(keys["gemini"], context, question, history)


# =============================================================================
# UI
# =============================================================================
st.title("🎧 Customer Support Bot")
st.caption("Retrieval-Augmented Generation over your support-ticket knowledge base.")

# ---- Model selector (sidebar) -------------------------------------------
provider = st.sidebar.selectbox("Choose LLM", ["Claude", "Gemini"])

# ---- API keys: secrets.toml first, sidebar fallback ----------------------
def get_key(secret_name, label):
    try:
        return st.secrets[secret_name]
    except Exception:
        return st.sidebar.text_input(label, type="password")

anthropic_key = get_key("ANTHROPIC_API_KEY", "Anthropic API key")
gemini_key = get_key("GEMINI_API_KEY", "Gemini API key")
keys = {"anthropic": anthropic_key, "gemini": gemini_key}
active_key = anthropic_key if provider == "Claude" else gemini_key

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Build (or load cached) the retrieval index
try:
    embedder, index, mini_docs = build_index()
    st.sidebar.success(f"Knowledge base ready: {len(mini_docs)} chunks indexed.")
except FileNotFoundError:
    st.error(f"Couldn't find {CSV_PATH}. Put the CSV in the same folder as this app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle a new question
if question := st.chat_input("Ask a support question..."):
    if not active_key:
        st.error(f"Add your {provider} API key in the sidebar to chat.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # ---- RETRIEVE + AUGMENT --------------------------------------------------
    hits = retrieve(embedder, index, mini_docs, question)
    context = build_context(hits)

    # ---- GENERATE (streamed) --------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            for chunk in stream_answer(provider, keys, context, question, history):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error calling {provider}: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ---- Show the retrieved sources (RAG transparency) -----------------------
    with st.expander("📄 Sources retrieved for this answer"):
        st.caption(f"Answered by: {provider}")
        for i, h in enumerate(hits, start=1):
            st.markdown(f"**[Doc {i}]** rid={h['rid']} · chunk={h['chunk_id']}")
            st.caption(h["text"][:300] + ("..." if len(h["text"]) > 300 else ""))