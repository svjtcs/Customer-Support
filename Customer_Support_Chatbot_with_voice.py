"""
==============================================================================
 RAG Customer Support Chatbot - Streamlit UI + Claude/Gemini + FREE Voice Chat
==============================================================================
 Voice chat with NO third paid API account. Speech-to-text and text-to-speech
 both run through services that cost nothing:
   - Speech-to-text: sent directly to GEMINI (you already have this key) -
     Gemini's regular API accepts audio as input, no separate STT service
     needed. Uses Gemini's free tier.
   - Text-to-speech: gTTS (Google Text-to-Speech) - a free Python library,
     NO API key or account required at all.

 HEADS-UP (unrelated to voice, but worth knowing): the `google.generativeai`
 package this app (and your original Claude/Gemini app) imports is now fully
 deprecated by Google - "no longer receiving updates or bug fixes." It still
 works today, which is why this script still uses it, but Google recommends
 migrating to the newer `google.genai` package when you get the chance. This
 is a separate, bigger change from adding voice, so it's flagged here rather
 than bundled in - happy to do that migration as its own step later.

 HONEST CAVEAT ON gTTS: it works by calling an UNOFFICIAL Google Translate
 endpoint (not a documented, supported API), so it's free but occasionally
 breaks if Google changes that endpoint without notice. Fine for a class
 project demo; not something to depend on for production.

 IMPORTANT: because transcription now goes through Gemini, your GEMINI_API_KEY
 is needed for voice input EVEN IF you pick Claude to answer the question.
 Claude still cannot accept audio input directly (no such API from Anthropic
 today) - it only ever sees the already-transcribed text, same as before.

 SETUP:
   pip install streamlit pandas sentence-transformers faiss-cpu anthropic \
               google-generativeai gtts

   Add to .streamlit/secrets.toml:
     ANTHROPIC_API_KEY = "sk-ant-..."
     GEMINI_API_KEY    = "AIza..."     # used for answering AND voice transcription

 RUN:
   streamlit run rag_streamlit_app_voice_free.py
==============================================================================
"""

import os
import re
import io
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic
import google.generativeai as genai
from gtts import gTTS

st.set_page_config(page_title="RAG Support Bot", page_icon="🎧", layout="centered")

CSV_PATH = "Customer_Support_Training_Dataset.csv"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_MODEL = "gemini-2.5-flash"

INDEX_FILE = "faiss.index"
DOCS_FILE = "mini_docs.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# =============================================================================
# STAGE 1-4: Load, clean, chunk, embed, and build the FAISS index.
# (Unchanged - voice doesn't touch retrieval.)
# =============================================================================
@st.cache_resource(show_spinner="Loading knowledge base and building search index...")
def build_index():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(INDEX_FILE)
        mini_docs = np.load(DOCS_FILE, allow_pickle=True).tolist()
        return embedder, index, mini_docs

    df = pd.read_csv(CSV_PATH)

    def clean(s):
        s = str(s)
        s = re.sub(r"\{\{.*?\}\}", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower()

    df["instruction_clean"] = df["instruction"].map(clean)
    df["response_clean"] = df["response"].map(clean)

    def chunk_words(text, n=200, overlap=30):
        words = text.split()
        if len(words) <= n:
            return [text]
        step = max(1, n - overlap)
        return [" ".join(words[i:i + n]) for i in range(0, len(words), step)]

    mini_docs = []
    for rid, row in df.iterrows():
        body = f"instruction: {row['instruction_clean']} | response: {row['response_clean']}"
        for j, ch in enumerate(chunk_words(body)):
            mini_docs.append({"rid": int(rid), "chunk_id": j, "text": ch})

    chunk_texts = [d["text"] for d in mini_docs]
    X = embedder.encode(
        chunk_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, INDEX_FILE)
    np.save(DOCS_FILE, np.array(mini_docs, dtype=object))

    return embedder, index, mini_docs


# =============================================================================
# VOICE (FREE): Gemini for speech-to-text, gTTS for text-to-speech
# =============================================================================
def transcribe_audio_with_gemini(gemini_key, audio_bytes):
    """Speech -> text, using Gemini's audio understanding (no separate STT
    service or key needed - reuses the same Gemini key used for answering)."""
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content([
        {"mime_type": "audio/wav", "data": audio_bytes},
        "Transcribe this audio exactly, word for word. "
        "Output ONLY the transcription, with no extra commentary.",
    ])
    return response.text.strip()


def synthesize_speech_free(text):
    """Text -> speech using gTTS. Free, no API key - but see the honest
    caveat at the top of this file about its unofficial nature."""
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# STAGE 5: RETRIEVE -> AUGMENT -> GENERATE (unchanged - Claude or Gemini)
# =============================================================================
def retrieve(embedder, index, mini_docs, question, k=3, pool=30):
    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    _, I = index.search(qv, pool)
    return [mini_docs[int(idx)] for idx in I[0][:k]]


def build_context(hits):
    return "\n\n".join([
        f"[Doc {i+1}] (rid={h['rid']}, chunk={h['chunk_id']})\n{h['text']}"
        for i, h in enumerate(hits)
    ])


SYSTEM_PROMPT = (
    "You are a helpful customer support assistant.\n"
    "Use only the provided context to answer the user's question.\n"
    "If the answer is not available in the context, say: "
    "'Sorry, I don't have that information.'\n"
    "Be polite, concise, and cite sources like [Doc 1], [Doc 2] when relevant.\n"
    "Keep answers short and conversational, since they may be read aloud."
)


def stream_claude_answer(api_key, context, question, history):
    client = Anthropic(api_key=api_key)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = history + [{"role": "user", "content": user_prompt}]
    with client.messages.stream(
        model=CLAUDE_MODEL, max_tokens=400, system=SYSTEM_PROMPT, messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_gemini_answer(api_key, context, question, history):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    gemini_history = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
        for m in history
    ]
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(user_prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_answer(provider, keys, context, question, history):
    if provider == "Claude":
        yield from stream_claude_answer(keys["anthropic"], context, question, history)
    elif provider == "Gemini":
        yield from stream_gemini_answer(keys["gemini"], context, question, history)


# =============================================================================
# UI
# =============================================================================
st.title("🎧 RAG Customer Support Bot")
st.caption("Retrieval-Augmented Generation over your support-ticket knowledge base. Free voice chat.")

provider = st.sidebar.selectbox("Choose LLM (for answering)", ["Claude", "Gemini"])
speak_replies = st.sidebar.checkbox("🔊 Speak the answer back (gTTS, free)", value=False)

def get_key(secret_name, label):
    try:
        return st.secrets[secret_name]
    except Exception:
        return st.sidebar.text_input(label, type="password")

anthropic_key = get_key("ANTHROPIC_API_KEY", "Anthropic API key")
gemini_key = get_key("GEMINI_API_KEY", "Gemini API key (also used for voice transcription)")
keys = {"anthropic": anthropic_key, "gemini": gemini_key}
active_key = anthropic_key if provider == "Claude" else gemini_key

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.pop("last_audio_id", None)
    st.rerun()

try:
    embedder, index, mini_docs = build_index()
    st.sidebar.success(f"Knowledge base ready: {len(mini_docs)} chunks indexed.")
except FileNotFoundError:
    st.error(f"Couldn't find {CSV_PATH}. Put the CSV in the same folder as this app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

# -----------------------------------------------------------------------------
# VOICE INPUT (free, via Gemini) - always needs a Gemini key, regardless of
# which provider answers the question.
# -----------------------------------------------------------------------------
st.markdown("**🎙️ Ask by voice** *(uses Gemini for transcription, free tier)*:")
audio_value = st.audio_input("Record your question")

typed_question = st.chat_input("...or type your question")

question = None
if audio_value is not None and audio_value.file_id != st.session_state.get("last_audio_id"):
    st.session_state.last_audio_id = audio_value.file_id
    if not gemini_key:
        st.error("Voice transcription uses Gemini - add your Gemini API key in the sidebar.")
        st.stop()
    with st.spinner("Transcribing..."):
        try:
            question = transcribe_audio_with_gemini(gemini_key, audio_value.getvalue())
            st.info(f"🎙️ Heard: \"{question}\"")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()
elif typed_question:
    question = typed_question

# -----------------------------------------------------------------------------
# Handle the question (voice-transcribed OR typed - identical from here on)
# -----------------------------------------------------------------------------
if question:
    if not active_key:
        st.error(f"Add your {provider} API key in the sidebar to chat.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    hits = retrieve(embedder, index, mini_docs, question)
    context = build_context(hits)

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

        answer_audio = None
        if speak_replies and full_response and not full_response.startswith("Error"):
            with st.spinner("Generating speech (gTTS)..."):
                try:
                    answer_audio = synthesize_speech_free(full_response)
                    st.audio(answer_audio, format="audio/mp3")
                except Exception as e:
                    st.warning(f"Couldn't generate speech (gTTS may be unavailable right now): {e}")

    st.session_state.messages.append({
        "role": "assistant", "content": full_response, "audio": answer_audio,
    })

    with st.expander("📄 Sources retrieved for this answer"):
        st.caption(f"Answered by: {provider}")
        for i, h in enumerate(hits, start=1):
            st.markdown(f"**[Doc {i}]** rid={h['rid']} · chunk={h['chunk_id']}")
            st.caption(h["text"][:300] + ("..." if len(h["text"]) > 300 else ""))
