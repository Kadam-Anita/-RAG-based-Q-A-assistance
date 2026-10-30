"""Core RAG logic for the Document Q&A app.

Ollama helpers, PDF indexing and the LangGraph pipeline live here, with no
Streamlit imports, so they can be tested or reused outside the UI.
"""

import json
import os
import re
import tempfile
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

# 127.0.0.1 instead of "localhost": on Windows "localhost" can resolve to IPv6 (::1)
# while Ollama only listens on IPv4, which gives WinError 10061.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

NO_ANSWER_MESSAGE = (
    "I couldn't find enough information about that in the uploaded documents. "
    "Try rephrasing the question, switching to **Fast** mode, retrieving more chunks, "
    "or lowering the relevance threshold in the sidebar."
)

ANSWER_MODES: Dict[str, Dict] = {
    "Fast": {
        "grade": False,
        "filter": False,
        "help": "Retrieve, then answer. Fewest model calls; best for small CPU servers.",
    },
    "Balanced": {
        "grade": True,
        "filter": False,
        "help": "Retrieve, grade each chunk for relevance, then answer.",
    },
    "Thorough": {
        "grade": True,
        "filter": True,
        "help": "Also filters context sentence by sentence. Most precise, slowest.",
    },
}


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    """Accept 'host:port' (what Render's `hostport` gives) as well as full URLs."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def auth_headers(api_key: str = "") -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _client_kwargs(api_key: str = "") -> Dict:
    return {"headers": auth_headers(api_key)} if api_key else {}


def list_models(base_url: str, api_key: str = "", timeout: float = 5) -> List[str]:
    """Return the names of models installed on the Ollama server."""
    resp = requests.get(
        f"{normalize_url(base_url)}/api/tags", headers=auth_headers(api_key), timeout=timeout
    )
    resp.raise_for_status()
    return [m.get("name") or m.get("model") for m in resp.json().get("models", [])]


def has_model(name: str, installed: Sequence[str]) -> bool:
    name = name.strip()
    if name in installed:
        return True
    return ":" not in name and f"{name}:latest" in installed


def pull_model(base_url: str, api_key: str, model: str) -> Iterator[dict]:
    """Download a model, yielding Ollama's progress events."""
    with requests.post(
        f"{normalize_url(base_url)}/api/pull",
        json={"model": model, "name": model, "stream": True},
        headers=auth_headers(api_key),
        stream=True,
        timeout=(10, 900),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(event["error"])
            yield event


def make_llm(base_url: str, api_key: str, model: str, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=model.strip(),
        base_url=normalize_url(base_url),
        temperature=temperature,
        num_ctx=4096,
        client_kwargs=_client_kwargs(api_key),
    )


def make_embeddings(base_url: str, api_key: str, model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model.strip(),
        base_url=normalize_url(base_url),
        client_kwargs=_client_kwargs(api_key),
    )


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def load_pdfs(files: Sequence[Tuple[str, bytes]]) -> List[Document]:
    """Load PDFs given as (filename, bytes). One Document per page."""
    documents: List[Document] = []
    for name, data in files:
        # PyPDFLoader needs a path, so write each upload to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            for page in PyPDFLoader(path).load():
                page.metadata["source"] = name
                documents.append(page)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return documents


def build_index(
    files: Sequence[Tuple[str, bytes]],
    embeddings,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    batch_size: int = 16,
    on_progress: Optional[Callable[[int, int], None]] = None,
):
    """Split PDFs into chunks and embed them in small batches into a FAISS index.

    Returns (vector_store, pages_with_text, chunk_count).
    """
    pages = [p for p in load_pdfs(files) if p.page_content.strip()]
    if not pages:
        raise ValueError("No extractable text found. Scanned/image-only PDFs need OCR first.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(pages)
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    store = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)
        if on_progress:
            on_progress(min(start + batch_size, len(chunks)), len(chunks))
    return store, len(pages), len(chunks)


# --------------------------------------------------------------------------- #
# Pipeline (retrieve -> grade -> refine), answer is streamed separately
# --------------------------------------------------------------------------- #
class State(TypedDict, total=False):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    scores: List[float]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    context: str


class DocEvalScore(BaseModel):
    score: float
    reason: str = ""


class KeepOrDrop(BaseModel):
    keep: bool


EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG. You will be given ONE retrieved "
     "chunk and a question. Return a relevance score between 0.0 and 1.0. "
     "1.0 means the chunk is sufficient to answer the question fully or mostly; "
     "0.0 means it is irrelevant. Be conservative with high scores. Give a short reason."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter. Return keep=true only if the sentence directly "
     "helps answer the question. Judge only the provided sentence."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful document Q&A assistant. Answer using ONLY the provided context. "
     "Be clear and concise, and use Markdown (lists, bold) when it helps. When the context "
     "labels its source, mention the file and page in brackets, e.g. [report.pdf, page 3]. "
     "If the context does not contain the answer, say that the uploaded documents do not "
     "contain enough information. Never invent facts."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def clamp_score(value) -> float:
    """Small models sometimes answer 85 instead of 0.85; normalise to [0, 1]."""
    value = float(value)
    if value > 1:
        value = value / 100 if value <= 100 else 1.0
    return max(0.0, min(1.0, value))


def format_docs(docs: Sequence[Document]) -> str:
    parts = []
    for d in docs:
        source = d.metadata.get("source", "document")
        page = d.metadata.get("page")
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source
        parts.append(f"[{label}]\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def split_sentences(text: str) -> List[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) > 20]


def build_graph(
    retriever,
    llm,
    *,
    grade: bool = True,
    filter_sentences: bool = False,
    upper: float = 0.7,
    lower: float = 0.3,
    max_sentences: int = 40,
):
    eval_chain = (
        EVAL_PROMPT | llm.with_structured_output(DocEvalScore, method="json_schema")
        if grade else None
    )
    filter_chain = (
        FILTER_PROMPT | llm.with_structured_output(KeepOrDrop, method="json_schema")
        if filter_sentences else None
    )

    def retrieve_node(state: State) -> dict:
        return {"docs": retriever.invoke(state["question"])}

    def grade_node(state: State) -> dict:
        docs = state.get("docs", [])
        if not grade:
            return {
                "good_docs": docs,
                "scores": [],
                "verdict": "NOT GRADED",
                "reason": "Chunk grading is off (Fast mode).",
            }

        scores: List[float] = []
        good: List[Document] = []
        for d in docs:
            try:
                score = clamp_score(
                    eval_chain.invoke({"question": state["question"], "chunk": d.page_content}).score
                )
            except Exception:
                # Model failed to return valid JSON: treat as borderline and keep the chunk.
                score = 0.5
            scores.append(score)
            if score > lower:
                good.append(d)

        if any(s > upper for s in scores):
            verdict, reason = "CORRECT", f"At least one chunk scored above {upper:.2f}."
        elif not good:
            verdict = "INCORRECT"
            reason = (
                "No chunks were retrieved."
                if not docs else f"All retrieved chunks scored at or below {lower:.2f}."
            )
        else:
            verdict, reason = "AMBIGUOUS", "Retrieved chunks have mixed or borderline relevance."
        return {"good_docs": good, "scores": scores, "verdict": verdict, "reason": reason}

    def refine_node(state: State) -> dict:
        good = state.get("good_docs", [])
        if not good:
            return {"strips": [], "kept_strips": [], "context": ""}
        if not filter_sentences:
            return {"strips": [], "kept_strips": [], "context": format_docs(good)}

        sentences = split_sentences("\n\n".join(d.page_content for d in good))
        kept: List[str] = []
        for sentence in sentences[:max_sentences]:
            try:
                if filter_chain.invoke({"question": state["question"], "sentence": sentence}).keep:
                    kept.append(sentence)
            except Exception:
                continue
        # If filtering dropped everything (or failed), fall back to the full chunks.
        context = "\n".join(kept).strip() or format_docs(good)
        return {"strips": sentences, "kept_strips": kept, "context": context}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("refine", refine_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "refine")
    graph.add_edge("refine", END)
    return graph.compile()


def stream_answer(llm, question: str, context: str) -> Iterator[str]:
    """Yield the answer token by token. Skips the model when there is no context."""
    if not context.strip():
        yield NO_ANSWER_MESSAGE
        return
    for chunk in (ANSWER_PROMPT | llm).stream({"question": question, "context": context}):
        text = chunk.content
        if isinstance(text, str) and text:
            yield text


def describe_sources(docs: Sequence[Document], scores: Sequence[float]) -> List[dict]:
    out = []
    for i, d in enumerate(docs):
        page = d.metadata.get("page")
        out.append({
            "source": d.metadata.get("source", "document"),
            "page": page + 1 if isinstance(page, int) else None,
            "excerpt": d.page_content.strip(),
            "score": scores[i] if i < len(scores) else None,
        })
    return out
