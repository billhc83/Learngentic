"""
Semantic similarity index for session prompts.

Uses a pure-Python TF-IDF implementation stored as JSON float arrays in the
sessions.pattern_embedding column. Cosine similarity computed in-memory over
at most a few hundred sessions, so no external libs needed.

The sqlite-vec virtual table is intentionally skipped — the corpus is small
enough that brute-force cosine in Python is under 5ms.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from learngentic.store.db import DEFAULT_DB_PATH, get_conn

# Stopwords: common English words that add no signal for dev prompt matching
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","be","as","at","this","that","have","had","was",
    "are","been","has","will","can","do","not","if","up","we","you","i",
    "me","my","your","our","its","their","there","here","so","just","then",
    "than","when","what","which","how","all","any","each","more","also",
    "would","could","should","may","might","need","use","get","set","let",
    "add","make","run","see","new","old","old","file","files","code",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF weights for a document's tokens given pre-computed IDF."""
    tf = Counter(tokens)
    n = len(tokens) or 1
    return {term: (count / n) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Build / refresh index
# ---------------------------------------------------------------------------

def build_index(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """
    Compute TF-IDF vectors for all sessions with prompts and store them
    in sessions.pattern_embedding. Returns {indexed: N, skipped: N}.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, initial_prompt FROM sessions "
            "WHERE initial_prompt IS NOT NULL AND initial_prompt != ''"
        ).fetchall()

    if not rows:
        return {"indexed": 0, "skipped": 0}

    # Compute IDF across all documents
    doc_tokens = [(r["session_id"], _tokenize(r["initial_prompt"][:2000])) for r in rows]
    N = len(doc_tokens)
    doc_freq: Counter = Counter()
    for _, tokens in doc_tokens:
        doc_freq.update(set(tokens))
    idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}

    # Compute and store vectors
    with get_conn(db_path) as conn:
        count = 0
        for session_id, tokens in doc_tokens:
            vec = _tfidf_vector(tokens, idf)
            conn.execute(
                "UPDATE sessions SET pattern_embedding = ? WHERE session_id = ?",
                (json.dumps(vec), session_id)
            )
            count += 1

    return {"indexed": count, "skipped": len(rows) - count}


def search_similar(
    prompt: str,
    k: int = 10,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Find k sessions most semantically similar to prompt.
    Returns list of dicts: {session_id, score} sorted descending.

    Uses the stored TF-IDF vectors. Falls back to empty list if no embeddings exist.
    """
    if not prompt or not prompt.strip():
        return []

    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, pattern_embedding FROM sessions "
            "WHERE pattern_embedding IS NOT NULL AND pattern_embedding != ''"
        ).fetchall()

    if not rows:
        return []

    # Build a corpus-level IDF from stored vectors to query against
    # (We re-use the stored vector keys as the vocabulary)
    vocab: set[str] = set()
    stored: list[tuple[str, dict[str, float]]] = []
    for row in rows:
        try:
            vec = json.loads(row["pattern_embedding"])
            stored.append((row["session_id"], vec))
            vocab.update(vec.keys())
        except (json.JSONDecodeError, TypeError):
            continue

    if not stored:
        return []

    # Build a simple IDF from stored vectors for the query
    N = len(stored)
    doc_freq: Counter = Counter()
    for _, vec in stored:
        doc_freq.update(vec.keys())
    idf = {term: math.log((N + 1) / (doc_freq.get(term, 0) + 1)) + 1 for term in vocab}

    # Embed the query using same IDF
    query_tokens = _tokenize(prompt[:2000])
    query_vec = _tfidf_vector(query_tokens, idf)

    # Score all sessions
    scored = [(sid, _cosine(query_vec, vec)) for sid, vec in stored]
    scored.sort(key=lambda x: -x[1])

    return [{"session_id": sid, "score": round(score, 4)} for sid, score in scored[:k] if score > 0]


def index_session(session_id: str, prompt: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Index a single new session. Rebuilds the full index so IDF stays correct.
    Lightweight enough for < 1000 sessions.
    """
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET pattern_embedding = NULL WHERE session_id = ?",
            (session_id,)
        )
    build_index(db_path)
