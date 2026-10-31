import uuid

import streamlit as st

from rag_graph import index_pdf, stream_answer

st.set_page_config(page_title="DocGraph", page_icon="📄", layout="wide")


# ---------- state helpers ----------
def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "doc_id": None, "doc_name": None, "messages": []}
    st.session_state.active = cid


def select_chat(cid: str):
    st.session_state.active = cid


def delete_chat(cid: str):
    st.session_state.chats.pop(cid, None)
    if not st.session_state.chats:
        new_chat()
    elif st.session_state.active == cid:
        st.session_state.active = next(reversed(st.session_state.chats))


if "chats" not in st.session_state:
    st.session_state.chats = {}      # chat_id -> {title, doc_id, doc_name, messages}
    st.session_state.docs = {}       # doc_id -> file name (PDFs already indexed this session)
    new_chat()

active = st.session_state.active
chat = st.session_state.chats[active]

# ---------- sidebar ----------
with st.sidebar:
    st.title("📄 DocGraph")
    st.button("➕ New chat", on_click=new_chat, use_container_width=True, type="primary")
    st.divider()
    for cid, c in reversed(list(st.session_state.chats.items())):
        label = ("🟢 " if cid == active else "💬 ") + c["title"]
        st.button(label, key=f"chat_{cid}", on_click=select_chat, args=(cid,), use_container_width=True)
    st.divider()
    st.button("🗑 Delete this chat", on_click=delete_chat, args=(active,), use_container_width=True)
    st.caption("LangGraph + Gemini embeddings + Groq (GPT-OSS 120B)")

# ---------- pick a PDF for this chat ----------
if chat["doc_id"] is None:
    st.subheader("Start a new chat")
    st.info("Upload a PDF to chat with it.")

    uploaded = st.file_uploader("PDF file", type="pdf", key=f"upload_{active}")
    if uploaded:
        error = None
        doc_id = None
        with st.status("Indexing PDF...", expanded=True) as status:
            bar = st.progress(0.0)

            def on_progress(msg: str, frac: float):
                status.update(label=msg)
                bar.progress(min(max(frac, 0.0), 1.0))

            try:
                doc_id = index_pdf(uploaded.getvalue(), progress=on_progress)
                status.update(label="Ready", state="complete")
            except Exception as e:
                error = str(e)
                status.update(label="Indexing failed", state="error")
        if error:
            st.error(error)
        else:
            st.session_state.docs[doc_id] = uploaded.name
            chat["doc_id"], chat["doc_name"] = doc_id, uploaded.name
            st.rerun()

    if st.session_state.docs:
        st.write("or reuse a PDF you already uploaded:")
        choice = st.selectbox(
            "Uploaded PDFs",
            options=list(st.session_state.docs.keys()),
            format_func=lambda d: st.session_state.docs[d],
            index=None,
            placeholder="Choose a PDF",
            label_visibility="collapsed",
            key=f"reuse_{active}",
        )
        if choice:
            chat["doc_id"], chat["doc_name"] = choice, st.session_state.docs[choice]
            st.rerun()
    st.stop()

# ---------- chat ----------
st.caption(f"📎 {chat['doc_name']}")

for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about the document..."):
    history = list(chat["messages"])          # previous turns, sent to the graph as memory
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box = st.empty()
        text, failed = "", None
        try:
            for ev in stream_answer(question, history, chat["doc_id"]):
                if ev["type"] == "reset":
                    text = ""
                    box.markdown("🔍 *Searching deeper...*")
                else:
                    text += ev["text"]
                    box.markdown(text + "▌")
            box.markdown(text)
        except Exception as e:
            failed = str(e)
            box.error(f"Something went wrong: {failed}")

    if not failed:
        chat["messages"] += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text},
        ]
        if chat["title"] == "New chat":
            chat["title"] = question[:28] + ("..." if len(question) > 28 else "")
        st.rerun()   # refresh sidebar title