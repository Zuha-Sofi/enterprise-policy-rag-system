import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    BM25_TOP_K,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    FINAL_TOP_N,
    HF_API_KEY,
    HF_MODEL,
    HIGH_SUPPORT_THRESHOLD,
    LLM_PROVIDER,
    LOW_SUPPORT_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    REQUEST_TIMEOUT,
    RERANK_TOP_N,
    RERANKER_MODEL,
    VECTOR_DIR,
)
from app.utils import (
    ensure_list_strings,
    extract_json_object,
    quote_supported,
    reciprocal_rank_fusion,
    shorten,
    sigmoid,
    simple_tokenize,
)


@dataclass
class RetrievalStore:
    embedder: SentenceTransformer
    reranker: CrossEncoder
    index: faiss.Index
    metadata: List[Dict]
    bm25: BM25Okapi


def _load_metadata(metadata_path: Path) -> List[Dict]:
    metadata: List[Dict] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    if not metadata:
        raise RuntimeError("Metadata file is empty.")
    return metadata


def load_store() -> RetrievalStore:
    index_path = VECTOR_DIR / "index.faiss"
    metadata_path = VECTOR_DIR / "metadata.jsonl"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Vector store not found in {VECTOR_DIR}. Run `python -m app.ingest --rebuild` first."
        )
    metadata = _load_metadata(metadata_path)
    bm25_corpus = [simple_tokenize(item["text"]) for item in metadata]

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    index = faiss.read_index(str(index_path))
    bm25 = BM25Okapi(bm25_corpus)

    return RetrievalStore(
        embedder=embedder,
        reranker=reranker,
        index=index,
        metadata=metadata,
        bm25=bm25,
    )


def dense_search(query: str, store: RetrievalStore, top_k: int = DENSE_TOP_K) -> List[Tuple[int, float]]:
    q = store.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = store.index.search(q, top_k)
    results = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx >= 0:
            results.append((idx, float(score)))
    return results


def bm25_search(query: str, store: RetrievalStore, top_k: int = BM25_TOP_K) -> List[Tuple[int, float]]:
    scores = store.bm25.get_scores(simple_tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in ranked.tolist()]


def hybrid_retrieve(query: str, store: RetrievalStore) -> List[Dict]:
    t0 = time.perf_counter()
    dense = dense_search(query, store)
    t1 = time.perf_counter()
    bm25 = bm25_search(query, store)
    t2 = time.perf_counter()

    dense_ids = [idx for idx, _ in dense]
    bm25_ids = [idx for idx, _ in bm25]
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[: max(RERANK_TOP_N, FINAL_TOP_N)]

    pairs = [(query, store.metadata[idx]["text"]) for idx in fused_ids]
    rerank_raw = store.reranker.predict(pairs).tolist() if pairs else []
    reranked = []
    for idx, raw in zip(fused_ids, rerank_raw):
        reranked.append((idx, float(raw), sigmoid(float(raw))))
    reranked.sort(key=lambda x: x[2], reverse=True)
    final = reranked[:FINAL_TOP_N]
    t3 = time.perf_counter()

    dense_lookup = {idx: score for idx, score in dense}
    bm25_lookup = {idx: score for idx, score in bm25}

    results: List[Dict] = []
    for rank, (idx, raw, prob) in enumerate(final, start=1):
        meta = dict(store.metadata[idx])
        meta["dense_score"] = dense_lookup.get(idx)
        meta["bm25_score"] = bm25_lookup.get(idx)
        meta["rerank_raw"] = raw
        meta["rerank_score"] = prob
        meta["rank"] = rank
        results.append(meta)

    timings = {
        "dense_seconds": round(t1 - t0, 4),
        "bm25_seconds": round(t2 - t1, 4),
        "rerank_seconds": round(t3 - t2, 4),
        "total_retrieval_seconds": round(t3 - t0, 4),
    }
    return [{"timings": timings}] + results


def build_prompt(question: str, chunks: List[Dict]) -> str:
    evidence_blocks = []
    for chunk in chunks:
        evidence_blocks.append(
            f"[{chunk['chunk_id']}] source={chunk['source']} page={chunk['page']}\n{chunk['text']}"
        )

    joined = "\n\n".join(evidence_blocks)
    return f"""
You are a secure internal policy assistant.

Follow these rules exactly:
1. Use only the evidence chunks below.
2. If the evidence does not clearly answer the question, return supported=false.
3. Do not use outside knowledge.
4. Keep the answer concise, precise, and professional.
5. Cite only chunk IDs that actually support the answer.

Return valid JSON only with this schema:
{{
  "answer": "string",
  "supported": true,
  "cited_chunk_ids": ["C0001"],
  "brief_quotes": ["short exact quote from evidence"],
  "confidence_label": "high|medium|low",
  "notes": "optional short note"
}}

Question:
{question}

Evidence:
{joined}
""".strip()


def _anthropic_client() -> Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError("Missing ANTHROPIC_API_KEY in environment.")
    return Anthropic(api_key=ANTHROPIC_API_KEY, timeout=REQUEST_TIMEOUT)


def _openai_client() -> OpenAI:
    if LLM_PROVIDER == "huggingface":
        if not HF_API_KEY:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in environment.")
        return OpenAI(
            api_key=HF_API_KEY,
            base_url="https://router.huggingface.co/v1",
            timeout=REQUEST_TIMEOUT,
        )
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "anthropic":
        client = _anthropic_client()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=900,
            temperature=0.1,
            system="You are a precise enterprise policy assistant that must return valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    client = _openai_client()
    model = OPENAI_MODEL if LLM_PROVIDER == "openai" else HF_MODEL
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {"role": "system", "content": "You are a precise enterprise policy assistant that must return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def validate_and_finalize(question: str, retrieved: List[Dict], llm_text: str) -> Dict:
    timings = retrieved[0]["timings"] if retrieved and "timings" in retrieved[0] else {}
    chunks = [item for item in retrieved if "chunk_id" in item]
    parsed = extract_json_object(llm_text)

    cited_ids = ensure_list_strings(parsed.get("cited_chunk_ids"))
    quotes = ensure_list_strings(parsed.get("brief_quotes"))
    cited_lookup = {c["chunk_id"]: c for c in chunks}

    valid_cited_ids = [cid for cid in cited_ids if cid in cited_lookup]
    max_rerank = max((c.get("rerank_score", 0.0) for c in chunks), default=0.0)
    top_chunk = chunks[0] if chunks else None

    quotes_ok = any(quote_supported(q, [cited_lookup[cid]["text"] for cid in valid_cited_ids]) for q in quotes)
    supported_flag = bool(parsed.get("supported")) and bool(valid_cited_ids)

    if not supported_flag or max_rerank < LOW_SUPPORT_THRESHOLD:
        answer = "I cannot find this in the policy documents."
        escalation_required = True
        confidence = "low"
        citations = []
    else:
        answer = (parsed.get("answer") or "").strip() or "I cannot find this in the policy documents."
        escalation_required = max_rerank < HIGH_SUPPORT_THRESHOLD or not quotes_ok
        confidence = parsed.get("confidence_label") if parsed.get("confidence_label") in {"high", "medium", "low"} else (
            "high" if max_rerank >= HIGH_SUPPORT_THRESHOLD and quotes_ok else "medium"
        )
        citations = [
            {
                "chunk_id": cid,
                "source": cited_lookup[cid]["source"],
                "page": cited_lookup[cid]["page"],
                "preview": cited_lookup[cid]["preview"],
                "rerank_score": round(cited_lookup[cid]["rerank_score"], 3),
            }
            for cid in valid_cited_ids
        ]

    return {
        "question": question,
        "answer": answer,
        "supported": answer != "I cannot find this in the policy documents.",
        "confidence_label": confidence,
        "top_rerank_score": round(max_rerank, 3),
        "escalation_required": escalation_required,
        "citations": citations,
        "retrieved_chunks": chunks,
        "timings": timings,
        "raw_model_output": llm_text,
        "debug": {
            "valid_cited_ids": valid_cited_ids,
            "quotes_ok": quotes_ok,
            "top_chunk": {
                "chunk_id": top_chunk["chunk_id"],
                "source": top_chunk["source"],
                "page": top_chunk["page"],
                "preview": shorten(top_chunk["text"], 180),
            } if top_chunk else None,
        },
    }


def answer_query(question: str, store: RetrievalStore) -> Dict:
    retrieved = hybrid_retrieve(question, store)
    chunks = [item for item in retrieved if "chunk_id" in item]
    prompt = build_prompt(question, chunks)

    start = time.perf_counter()
    llm_text = call_llm(prompt)
    generation_seconds = time.perf_counter() - start

    result = validate_and_finalize(question, retrieved, llm_text)
    result["timings"]["generation_seconds"] = round(generation_seconds, 4)
    result["timings"]["total_seconds"] = round(
        result["timings"].get("total_retrieval_seconds", 0.0) + generation_seconds, 4
    )
    return result
