"""Document Q&A: Streamlit UI.

Run locally:  streamlit run app.py
Render / Streamlit Community Cloud: see README.md
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    ANSWER_MODES,
    DEFAULT_BASE_URL,
    build_graph,
    build_index,
    describe_sources,
    has_model,
    list_models,
    make_embeddings,
    make_llm,
    normalize_url,
    pull_model,
    stream_answer,
)

load_dotenv()
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


def get_setting(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets, then the default."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets[name])
    except Exception:
        return default


ss = st.session_state
ss.setdefault("messages", [])        # chat history
ss.setdefault("index", None)         # dict with the FAISS store and stats
ss.setdefault("installed", None)     # models installed on the Ollama server
ss.setdefault("conn_error", None)
ss.setdefault("auto_checked", False)


# --------------------------------------------------------------------------- #
# Sidebar helpers
# --------------------------------------------------------------------------- #
def refresh_models(base_url: str, api_key: str) -> None:
    try:
        with st.spinner("Checking Ollama..."):
            ss.installed = list_models(base_url, api_key)
        ss.conn_error = None
    except Exception as exc:
        ss.installed = None
        ss.conn_error = f"Could not reach Ollama at {normalize_url(base_url)} ({type(exc).__name__})."


def download_models(base_url: str, api_key: str, missing: list) -> None:
    for model in missing:
        bar = st.progress(0.0, text=f"Downloading {model}...")
        fraction = 0.0
        try:
            for event in pull_model(base_url, api_key, model):
                if event.get("total") and event.get("completed") is not None:
                    fraction = min(event["completed"] / event["total"], 1.0)
                bar.progress(fraction, text=f"{model}: {event.get('status', 'downloading')}")
        except Exception as exc:
            bar.empty()
            st.error(f"Could not download {model}: {exc}")
            return
        bar.progress(1.0, text=f"{model} is ready")
    refresh_models(base_url, api_key)
    st.rerun()


def connection_panel(base_url: str, api_key: str, models: list) -> None:
    if st.button("Test connection", use_container_width=True):
        refresh_models(base_url, api_key)

    if ss.conn_error:
        st.error(ss.conn_error)
        st.caption("Is Ollama running? On Render, is the `ollama` service live?")
        return
    if ss.installed is None:
        return

    st.success(f"Connected · {len(ss.installed)} model(s) installed")
    missing = [m.strip() for m in models if m.strip() and not has_model(m, ss.installed)]
    if missing:
        st.warning("Not installed yet: " + ", ".join(f"`{m}`" for m in missing))
        if st.button("Download missing models", use_container_width=True):
            download_models(base_url, api_key, missing)


def run_build(uploaded, base_url: str, api_key: str, embed_model: str, chunk_size: int) -> None:
    files = [(f.name, f.getvalue()) for f in uploaded]
    bar = st.progress(0.0, text="Reading PDFs...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Embedding chunks {done}/{total}")

    try:
        store, pages, chunks = build_index(
            files,
            make_embeddings(base_url, api_key, embed_model),
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 6,
            on_progress=on_progress,
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Could not build the index: {exc}")
        st.caption("Check that Ollama is reachable and the embedding model is installed.")
        return

    bar.empty()
    ss.index = {
        "store": store,
        "pages": pages,
        "chunks": chunks,
        "files": [name for name, _ in files],
        "embed_model": embed_model.strip(),
    }
    ss.messages = []  # a new index starts a new conversation


# --------------------------------------------------------------------------- #
# Chat helpers
# --------------------------------------------------------------------------- #
STEP_LABELS = {
    "retrieve": "Checking which passages are relevant...",
    "grade": "Selecting the best context...",
    "refine": "Context ready",
}


def render_trace(trace: dict) -> None:
    n = len(trace["sources"])
    title = f"Sources & details · {n} passage{'s' if n != 1 else ''} · {trace['elapsed']:.1f}s"
    with st.expander(title):
        st.markdown(
            f"**Mode:** {trace['mode']}  \n"
            f"**Relevance check:** {trace['verdict']}. {trace['reason']}"
        )
        if trace.get("total_sentences"):
            st.markdown(f"**Sentences kept:** {trace['kept_sentences']} of {trace['total_sentences']}")
        for i, src in enumerate(trace["sources"], start=1):
            page = f" · page {src['page']}" if src["page"] else ""
            score = f" · relevance {src['score']:.2f}" if src["score"] is not None else ""
            st.markdown(f"**{i}. {src['source']}{page}{score}**")
            st.caption(" ".join(src["excerpt"].split())[:600].replace("$", "\\$"))


def answer_question(question: str, cfg: dict) -> None:
    started = time.perf_counter()
    try:
        llm = make_llm(cfg["base_url"], cfg["api_key"], cfg["llm_model"])
        retriever = ss.index["store"].as_retriever(
            search_type="similarity", search_kwargs={"k": cfg["top_k"]}
        )
        graph = build_graph(
            retriever,
            llm,
            grade=cfg["mode"]["grade"],
            filter_sentences=cfg["mode"]["filter"],
            upper=cfg["upper"],
            lower=cfg["lower"],
        )

        state = {"question": question}
        with st.status("Searching your documents...", expanded=False) as status:
            for update in graph.stream({"question": question}, stream_mode="updates"):
                for node, output in update.items():
                    state.update(output)
                    status.update(label=STEP_LABELS.get(node, node))
            kept = len(state.get("good_docs", []))
            status.update(label=f"Using {kept} passage{'s' if kept != 1 else ''}", state="complete")

        answer = st.write_stream(stream_answer(llm, question, state.get("context", "")))
    except Exception as exc:
        message = (
            f"⚠️ Something went wrong: {exc}\n\n"
            "Use **Test connection** in the sidebar to check that Ollama is reachable "
            "and that both models are installed."
        )
        st.error(message)
        ss.messages.append({"role": "assistant", "content": message})
        return

    trace = {
        "mode": cfg["mode_name"],
        "verdict": state.get("verdict", ""),
        "reason": state.get("reason", ""),
        "sources": describe_sources(state.get("docs", []), state.get("scores", [])),
        "kept_sentences": len(state.get("kept_strips", [])),
        "total_sentences": len(state.get("strips", [])),
        "elapsed": time.perf_counter() - started,
    }
    render_trace(trace)
    ss.messages.append({"role": "assistant", "content": answer, "trace": trace})


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📚 Document Q&A")

    st.subheader("1 · Ollama")
    base_url = st.text_input(
        "Ollama URL",
        value=get_setting("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        help="Local: http://127.0.0.1:11434. On Render this is filled in automatically.",
    )
    api_key = st.text_input(
        "API key (optional)",
        value=get_setting("OLLAMA_API_KEY"),
        type="password",
        help="Sent as a Bearer token. Only needed if your Ollama sits behind an authenticated proxy.",
    )
    llm_model = st.text_input("Chat model", value=get_setting("OLLAMA_MODEL", "gemma3:1b"))
    embed_model = st.text_input(
        "Embedding model", value=get_setting("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    if not ss.auto_checked:
        ss.auto_checked = True
        refresh_models(base_url, api_key)
    connection_panel(base_url, api_key, [llm_model, embed_model])

    st.divider()
    st.subheader("2 · Documents")
    uploaded = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    st.divider()
    st.subheader("3 · Answer settings")
    mode_name = st.radio("Answer mode", list(ANSWER_MODES), index=0)
    st.caption(ANSWER_MODES[mode_name]["help"])
    with st.expander("Advanced"):
        top_k = st.slider("Chunks to retrieve", 1, 8, 4)
        chunk_size = st.select_slider(
            "Chunk size (characters, used on next index build)",
            options=[400, 600, 800, 1000, 1200, 1500],
            value=800,
        )
        upper = st.slider("Highly relevant above", 0.5, 1.0, 0.7, 0.05)
        lower = st.slider("Keep chunks scoring above", 0.0, 0.6, 0.3, 0.05)

    if st.button("Build index", type="primary", use_container_width=True, disabled=not uploaded):
        run_build(uploaded, base_url, api_key, embed_model, chunk_size)

    if ss.messages and st.button("Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("📚 Document Q&A")
st.caption("Ask questions about your PDFs. Answers are generated by your Ollama model and grounded in your documents.")

index = ss.index
if index is None:
    steps = [
        ("🔌", "Connect", "Enter your Ollama URL and models in the sidebar, then check the connection."),
        ("📄", "Upload", "Add one or more text-based PDFs and click **Build index**."),
        ("💬", "Ask", "Type a question below. Every answer shows the passages it came from."),
    ]
    for column, (icon, title, text) in zip(st.columns(3), steps):
        with column, st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(text)
    st.stop()

st.caption(
    f"Indexed **{', '.join(index['files'])}** · {index['pages']} pages · "
    f"{index['chunks']} chunks · embeddings: `{index['embed_model']}`"
)

for message in ss.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            render_trace(message["trace"])

question = st.chat_input("Ask something about your documents...")
if question:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer_question(
            question,
            {
                "base_url": base_url,
                "api_key": api_key,
                "llm_model": llm_model,
                "top_k": top_k,
                "upper": upper,
                "lower": lower,
                "mode_name": mode_name,
                "mode": ANSWER_MODES[mode_name],
            },
        )
