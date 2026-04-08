import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    MIN_CHUNK_CHARS,
    VECTOR_DIR,
)
from utils import shorten


def load_documents() -> List[Dict]:
    documents = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.replace("\x00", " ").strip()
            if text:
                documents.append(
                    {
                        "source": pdf_path.name,
                        "page": page_idx,
                        "text": text,
                    }
                )
    return documents


def chunk_text(text: str, tokenizer, target_tokens: int, overlap_tokens: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    def token_len(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    for para in paragraphs:
        para_tokens = token_len(para)
        if para_tokens > target_tokens * 1.35:
            sentences = [s.strip() for s in para.replace("\n", " ").split(". ") if s.strip()]
            for sentence in sentences:
                s_tokens = token_len(sentence)
                if current_parts and current_tokens + s_tokens > target_tokens:
                    chunks.append("\n\n".join(current_parts).strip())
                    overlap_parts = []
                    overlap_count = 0
                    for prev in reversed(current_parts):
                        prev_tokens = token_len(prev)
                        if overlap_count + prev_tokens > overlap_tokens:
                            break
                        overlap_parts.insert(0, prev)
                        overlap_count += prev_tokens
                    current_parts = overlap_parts
                    current_tokens = overlap_count
                current_parts.append(sentence if sentence.endswith(".") else sentence + ".")
                current_tokens += s_tokens
            continue

        if current_parts and current_tokens + para_tokens > target_tokens:
            chunks.append("\n\n".join(current_parts).strip())
            overlap_parts = []
            overlap_count = 0
            for prev in reversed(current_parts):
                prev_tokens = token_len(prev)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_parts.insert(0, prev)
                overlap_count += prev_tokens
            current_parts = overlap_parts
            current_tokens = overlap_count

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def build_chunks(documents: List[Dict]) -> List[Dict]:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    chunks: List[Dict] = []
    chunk_id = 0
    for doc in documents:
        for chunk_text_value in chunk_text(
            doc["text"], tokenizer=tokenizer, target_tokens=CHUNK_SIZE, overlap_tokens=CHUNK_OVERLAP
        ):
            chunk_id += 1
            chunks.append(
                {
                    "chunk_id": f"C{chunk_id:04d}",
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk_text_value,
                    "preview": shorten(chunk_text_value, 220),
                }
            )
    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")
    return chunks


def build_faiss_index(chunks: List[Dict]) -> None: 
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

    with open(VECTOR_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    stats = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "dimension": int(embeddings.shape[1]),
    }
    with open(VECTOR_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main(rebuild: bool = False):
    start = time.perf_counter()
    if rebuild and VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()
    chunks = build_chunks(documents)
    build_faiss_index(chunks)

    elapsed = time.perf_counter() - start
    print(f"Loaded {len(documents)} pages")
    print(f"Built {len(chunks)} chunks")
    print(f"Saved index to {VECTOR_DIR}")
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Secure Policy RAG vector index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the vector store.")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
