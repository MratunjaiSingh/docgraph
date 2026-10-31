"""DocGraph core: PDF -> chunks -> embeddings -> LangGraph RAG with chat memory + streaming."""
import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

# ---------- config ----------
INDEX_DIR = Path("indexes")
INDEX_VERSION = "2"   # bump when chunking/extraction changes, so old caches are ignored
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH = 50      # free tier: 100 embed requests/min
EMBED_WAIT = 65       # seconds to wait after a 429
DEFAULT_K = 8
RETRY_K = 12          # widened search if the first answer is "I dont know"
HISTORY_TURNS = 6     # last N messages sent to the model
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "5"))      # protects API quota on a public deploy
MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))

embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
llm = ChatGroq(model=LLM_MODEL, temperature=0)

_STORES: dict[str, InMemoryVectorStore] = {}   # doc_id -> store (in-memory cache)


# ---------- indexing ----------
def _extract_links(reader: PdfReader) -> list[str]:
    """Clickable links hide their URL in annotations, extract_text() does not return them."""
    links: list[str] = []
    for page in reader.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        for annot in annots.get_object():
            try:
                action = annot.get_object().get("/A")
                uri = action.get_object().get("/URI") if action is not None else None
            except Exception:
                continue
            if uri and str(uri) not in links:
                links.append(str(uri))
    return links


def _pdf_to_chunks(pdf_bytes: bytes) -> list[Document]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDF has {len(reader.pages)} pages, limit is {MAX_PAGES}.")
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"page": i + 1}))
    if not pages:
        raise ValueError("No text found in this PDF (scanned/image-only PDFs are not supported).")
    links = _extract_links(reader)
    if links:
        body = "Hyperlinks (URLs) in this document - GitHub, LinkedIn, live demo / deploy links, email, website:\n"
        pages.append(Document(page_content=body + "\n".join(f"- {u}" for u in links), metadata={"page": 0}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)


def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


def index_pdf(pdf_bytes: bytes, progress: Optional[Callable[[str, float], None]] = None) -> str:
    """Embed a PDF (batched, rate-limit safe, cached on disk). Returns doc_id."""
    report = progress or (lambda msg, frac: print(msg))
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"PDF is larger than {MAX_PDF_MB} MB.")
    doc_id = hashlib.sha256(INDEX_VERSION.encode() + pdf_bytes).hexdigest()[:16]

    if doc_id in _STORES:
        return doc_id

    path = INDEX_DIR / f"{doc_id}.json"
    if path.exists():
        report("Loading cached index...", 1.0)
        _STORES[doc_id] = InMemoryVectorStore.load(str(path), embeddings)
        return doc_id

    chunks = _pdf_to_chunks(pdf_bytes)
    total = len(chunks)
    report(f"{total} chunks created, embedding...", 0.0)

    store = InMemoryVectorStore(embeddings)
    for i in range(0, total, EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        for attempt in range(5):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                if _is_rate_limit(e) and attempt < 4:
                    report(f"Rate limit hit, waiting {EMBED_WAIT}s...", i / total)
                    time.sleep(EMBED_WAIT)
                else:
                    raise
        done = min(i + EMBED_BATCH, total)
        report(f"Embedded {done}/{total} chunks", done / total)

    INDEX_DIR.mkdir(exist_ok=True)
    store.dump(str(path))
    _STORES[doc_id] = store
    return doc_id


def get_store(doc_id: str) -> InMemoryVectorStore:
    if doc_id not in _STORES:
        path = INDEX_DIR / f"{doc_id}.json"
        if not path.exists():
            raise KeyError(f"Index for document {doc_id} not found. Upload the PDF again.")
        _STORES[doc_id] = InMemoryVectorStore.load(str(path), embeddings)
    return _STORES[doc_id]


# ---------- language ----------
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_IDK = {"en": "I dont know.", "hi": "\u092e\u0941\u091d\u0947 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e\u0964"}   # "मुझे नहीं पता।"
_IDK_MARKERS = ("i dont know", "\u092e\u0941\u091d\u0947 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e")
_LANG_RULE = {
    "en": "Reply in English.",
    "hi": "Reply in Hindi, written in Devanagari script. Keep names, technical terms, code identifiers "
          "and URLs exactly as they appear in the Context (do not translate or transliterate them).",
}


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


# ---------- state ----------
class RagState(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)   # [{"role": "user"|"assistant", "content": str}]
    standalone: str = ""
    documents: list = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    k: int = DEFAULT_K
    retried: bool = False
    answer_lang: str = "en"          # "en" or "hi"


def _format_history(history: list[dict]) -> str:
    lines = []
    for m in history[-HISTORY_TURNS:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:500]}")
    return "\n".join(lines)


# ---------- nodes ----------
def condense_node(state: RagState) -> dict:
    """Rewrite follow-ups into standalone English questions so retrieval works."""
    if not state.history and not _has_devanagari(state.question):
        return {"standalone": state.question}
    prompt = f"""Rewrite the follow-up question below as a standalone question in English that makes sense
without the conversation (translate it to English if it is written in another language).
If it is already a standalone English question, return it unchanged. Return ONLY the question.

Conversation:
{_format_history(state.history) or "(none)"}

Follow-up question: {state.question}"""
    out = str(llm.invoke(prompt).content).strip()
    return {"standalone": out or state.question}


def retrieve_node(state: RagState, config: RunnableConfig) -> dict:
    store = get_store(config["configurable"]["doc_id"])
    docs = store.similarity_search(query=state.standalone or state.question, k=state.k)
    return {"documents": docs}


def create_context_node(state: RagState) -> dict:
    return {"context": "\n\n".join(d.page_content for d in state.documents)}


def generate_node(state: RagState) -> dict:
    history_block = _format_history(state.history) or "(none)"
    lang = state.answer_lang if state.answer_lang in _LANG_RULE else "en"
    prompt = f"""
You are an assistant answering questions about a document.
Rules:
- Facts about the document must come ONLY from the Context. Do not infer, assume, or exaggerate.
- Use the Conversation only to understand follow-up questions or to answer questions about the conversation itself.
- Copy names exactly as written in the Context.
- {_LANG_RULE[lang]}
- If the answer is not in the Context (or the Conversation), reply exactly: {_IDK[lang]}

Conversation:
{history_block}

Context:
{state.context}

Question: {state.question}
"""
    response = llm.invoke(prompt)
    return {"answer": str(response.content)}


def widen_search_node(state: RagState) -> dict:
    return {"k": RETRY_K, "retried": True}


# ---------- conditional edge ----------
def route_after_generate(state: RagState) -> str:
    answer = state.answer.lower().replace("'", "").replace("\u2019", "")
    said_idk = any(marker in answer for marker in _IDK_MARKERS)
    if said_idk and not state.retried:
        return "widen_search_node"
    return END


# ---------- graph ----------
graph = StateGraph(RagState)
graph.add_node("condense_node", condense_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("create_context_node", create_context_node)
graph.add_node("generate_node", generate_node)
graph.add_node("widen_search_node", widen_search_node)

graph.add_edge(START, "condense_node")
graph.add_edge("condense_node", "retrieve_node")
graph.add_edge("retrieve_node", "create_context_node")
graph.add_edge("create_context_node", "generate_node")
graph.add_conditional_edges("generate_node", route_after_generate, ["widen_search_node", END])
graph.add_edge("widen_search_node", "retrieve_node")

app = graph.compile()


# ---------- public API ----------
def stream_answer(question: str, history: list[dict], doc_id: str, answer_lang: str = "auto") -> Iterator[dict]:
    """answer_lang: "auto" (match the question), "en" or "hi".
    Yield {"type": "token", "text": str} as the answer streams.
    Yields {"type": "reset"} if the retry edge fires and the answer restarts."""
    if answer_lang == "auto":      # match the language the question was asked in
        answer_lang = "hi" if _has_devanagari(question) else "en"
    last_step = None
    for chunk, meta in app.stream(
        {"question": question, "history": history, "answer_lang": answer_lang},
        config={"configurable": {"doc_id": doc_id}},
        stream_mode="messages",
    ):
        if meta.get("langgraph_node") != "generate_node":
            continue
        text = chunk.content
        if not isinstance(text, str) or not text:
            continue
        step = meta.get("langgraph_step")
        if last_step is not None and step != last_step:
            yield {"type": "reset"}
        last_step = step
        yield {"type": "token", "text": text}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python rag_graph.py path/to/file.pdf")
    doc_id = index_pdf(Path(sys.argv[1]).read_bytes())
    chat_history: list[dict] = []
    while True:
        q = input("\nQuestion (q to quit): ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        answer = ""
        for ev in stream_answer(q, chat_history, doc_id):
            if ev["type"] == "reset":
                answer = ""
                print("\n[searching deeper...]")
            else:
                answer += ev["text"]
                print(ev["text"], end="", flush=True)
        print()
        chat_history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]