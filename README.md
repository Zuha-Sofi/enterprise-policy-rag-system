# Enterprise Policy RAG - Advanced Document Intelligence System

An end-to-end Retrieval-Augmented Generation (RAG) system designed to answer policy-related questions with high accuracy using hybrid retrieval, reranking, and grounded LLM responses.

This project simulates real-world enterprise document intelligence systems used in compliance, legal, and HR domains.

## Architecture

User Query  
→ Dense Retrieval (FAISS)  
→ BM25 Retrieval  
→ Reciprocal Rank Fusion (RRF)  
→ Cross-Encoder Reranking  
→ Top-K Context Selection  
→ LLM (Anthropic / OpenAI)  
→ JSON Output with Citations  
→ Answer Validation + Safe Fallback

## Architecture
![Architecture](docs/architecture.png)

## Key Features

- Hybrid retrieval combining semantic search (embeddings) and lexical search (BM25)
- Reciprocal Rank Fusion (RRF) to improve retrieval robustness
- Cross-encoder reranking for high-precision context selection
- Grounded LLM responses with explicit source citations
- Structured JSON outputs for reliability and downstream use
- Safe fallback mechanism when evidence is insufficient
- Evaluation pipeline to measure retrieval and answer quality
- Benchmarking tools to analyze latency and performance

## Tech Stack

- Python
- FAISS (Vector Database)
- Sentence Transformers (Embeddings)
- BM25 (Lexical Retrieval)
- Cross-Encoder Models (Reranking)
- Anthropic / OpenAI (LLMs)
- Streamlit (UI)

## How It Works

1. Documents are ingested, chunked, and converted into embeddings.
2. Queries are processed using hybrid retrieval (FAISS + BM25).
3. Results are fused and reranked using a cross-encoder.
4. Top-ranked context is passed to the LLM.
5. The system generates a grounded response with citations.
6. If confidence is low, a safe fallback response is returned.

## Data Disclaimer

The original policy documents used in this project are not included due to confidentiality.

Sample documents are provided for demonstration purposes. Users can upload their own documents to test the system.
