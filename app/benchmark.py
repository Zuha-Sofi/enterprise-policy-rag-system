import argparse
import json
import statistics
import time

from app.config import EVAL_DIR
from app.rag_pipeline import answer_query, hybrid_retrieve, load_store


def load_queries():
    with open(EVAL_DIR / "qa_eval_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    return [case["question"] for case in cases]


def main(iterations: int = 1, retrieval_only: bool = False):
    store = load_store()
    queries = load_queries()

    retrieval_times = []
    total_times = []

    for _ in range(iterations):
        for query in queries:
            start = time.perf_counter()
            if retrieval_only:
                retrieved = hybrid_retrieve(query, store)
                elapsed = time.perf_counter() - start
                retrieval_times.append(elapsed)
                print(f"{query} -> retrieval {elapsed:.3f}s")
                continue

            result = answer_query(query, store)
            elapsed = time.perf_counter() - start
            retrieval_times.append(result["timings"].get("total_retrieval_seconds", 0.0))
            total_times.append(elapsed)
            print(f"{query} -> total {elapsed:.3f}s")

    print("\nBenchmark summary")
    print(f"queries: {len(queries) * iterations}")
    if retrieval_times:
        print(f"avg retrieval: {statistics.mean(retrieval_times):.3f}s")
        print(f"p95 retrieval: {sorted(retrieval_times)[max(0, int(len(retrieval_times) * 0.95) - 1)]:.3f}s")
    if total_times:
        print(f"avg total: {statistics.mean(total_times):.3f}s")
        print(f"p95 total: {sorted(total_times)[max(0, int(len(total_times) * 0.95) - 1)]:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Secure Policy RAG.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    main(iterations=args.iterations, retrieval_only=args.retrieval_only)
