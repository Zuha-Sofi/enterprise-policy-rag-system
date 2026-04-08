import streamlit as st

from rag_pipeline import answer_query, load_store

st.set_page_config(page_title="Enterprice Policy Assistant", layout="wide")
st.title("🔐 Secure Policy Intelligence Assistant")
st.caption("Hybrid retrieval + reranking + grounded citations")

@st.cache_resource
def initialize_store():
    return load_store()

try:
    store = initialize_store()
except Exception as exc:
    st.error(f"Startup error: {exc}")
    st.stop()

with st.sidebar:
    st.header("How this version is different")
    st.write("- Dense retrieval + BM25")
    st.write("- Cross-encoder reranking")
    st.write("- Strict grounded JSON output")
    st.write("- Citation validation and safe fallback")
    st.write("- Evaluation and benchmark scripts included")

    st.header("Sample prompts")
    st.code(
        "\n".join(
            [
                "Can a whistleblower remain anonymous?",
                "Is a facilitation payment allowed?",
                "Who does the Diversity & Inclusion policy apply to?",
                "What does the Privacy Policy say about security of personal information?",
            ]
        )
    )

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a policy question...")
if query:
    with st.spinner("Retrieving evidence and generating answer..."):
        result = answer_query(query, store)

    st.session_state.history.append(("user", query, None))
    st.session_state.history.append(("assistant", result["answer"], result))

for role, text, payload in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if role == "assistant" and payload:
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", payload["confidence_label"].upper())
            c2.metric("Top evidence", f'{payload["top_rerank_score"]:.3f}')
            c3.metric("Latency", f'{payload["timings"].get("total_seconds", 0.0):.2f}s')

            if payload["escalation_required"]:
                st.warning("Low support or weak evidence - treat this answer carefully or escalate.")
            else:
                st.success("Grounded answer with validated evidence.")

            st.markdown("#### Citations")
            if payload["citations"]:
                for citation in payload["citations"]:
                    st.write(
                        f"- **{citation['chunk_id']}** | {citation['source']} p.{citation['page']} | score={citation['rerank_score']}"
                    )
                    st.caption(citation["preview"])
            else:
                st.write("No reliable citations available.")

            with st.expander("Retrieved evidence"):
                for chunk in payload["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['chunk_id']}** - {chunk['source']} p.{chunk['page']} - rerank={chunk['rerank_score']:.3f}"
                    )
                    st.write(chunk["text"])

            with st.expander("Timings"):
                st.json(payload["timings"])
