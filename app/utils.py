import json
import math
import re
from typing import Any, Dict, Iterable, List


WORD_RE = re.compile(r"[A-Za-z0-9_\-']+")


def slugify_filename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def simple_tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return {}
    candidate = text[first:last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def ensure_list_strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def shorten(text: str, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def quote_supported(quote: str, chunks: Iterable[str]) -> bool:
    q = re.sub(r"\s+", " ", (quote or "")).strip().lower()
    if not q:
        return False
    for chunk in chunks:
        chunk_norm = re.sub(r"\s+", " ", (chunk or "")).strip().lower()
        if q in chunk_norm:
            return True
    return False
