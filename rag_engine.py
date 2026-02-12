from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests

from config import (
    CHROMA_DIR,
    CORPUS_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source: str


def _strip_html(raw: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_text(raw: str, suffix: str) -> str:
    if suffix.lower() in {".html", ".htm"}:
        return _strip_html(raw)
    return re.sub(r"\s+", " ", raw).strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    clean = text.strip()
    if not clean:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        part = clean[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(clean):
            break
        start += step

    return chunks


def _hash_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def load_corpus_chunks(corpus_dir: str | Path = CORPUS_DIR, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[ChunkRecord]:
    base = Path(corpus_dir)
    if not base.exists():
        return []

    allowed = {".txt", ".md", ".html", ".htm"}
    records: list[ChunkRecord] = []

    for file_path in sorted(base.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in allowed:
            continue
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        normalized = _normalize_text(raw, file_path.suffix)
        if not normalized:
            continue

        rel_source = str(file_path.relative_to(base))
        for idx, part in enumerate(chunk_text(normalized, chunk_size=chunk_size, overlap=overlap), start=1):
            records.append(
                ChunkRecord(
                    chunk_id=_hash_id(rel_source, idx, part),
                    text=part,
                    source=rel_source,
                )
            )

    return records


def _ollama_embeddings(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        vectors.append(payload["embedding"])
    return vectors


def _ollama_chat(messages: list[dict[str, str]], model: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return (payload.get("message", {}) or {}).get("content", "").strip()


def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(name: str = DEFAULT_COLLECTION):
    client = _client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def reset_collection(name: str = DEFAULT_COLLECTION) -> None:
    client = _client()
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)


def train_collection(
    corpus_dir: str | Path = CORPUS_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = True,
) -> dict[str, Any]:
    chunks = load_corpus_chunks(corpus_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {
            "ok": False,
            "error": "No se encontraron documentos válidos en el corpus (.txt/.md/.html).",
        }

    if reset:
        reset_collection(collection_name)

    collection = get_or_create_collection(collection_name)

    docs = [row.text for row in chunks]
    ids = [row.chunk_id for row in chunks]
    metas = [{"source": row.source} for row in chunks]

    embeddings = _ollama_embeddings(docs, model=embed_model)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    return {
        "ok": True,
        "collection": collection_name,
        "documents": len(set(row.source for row in chunks)),
        "chunks": len(chunks),
        "embed_model": embed_model,
    }


def semantic_search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    top_k: int = 4,
) -> dict[str, Any]:
    text = query.strip()
    if not text:
        return {"ok": False, "error": "La consulta está vacía."}

    collection = get_or_create_collection(collection_name)
    if collection.count() == 0:
        return {"ok": False, "error": "La colección está vacía. Entrena primero el corpus."}

    q_vector = _ollama_embeddings([text], model=embed_model)[0]
    result = collection.query(query_embeddings=[q_vector], n_results=top_k)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    items: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        distance = float(dists[idx]) if idx < len(dists) else 1.0
        score = max(0.0, 1.0 - distance)
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        items.append(
            {
                "rank": idx + 1,
                "score": round(score, 4),
                "source": meta.get("source", "desconocido"),
                "text": doc,
            }
        )

    return {"ok": True, "query": text, "results": items}


def rag_answer(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    chat_model: str = "qwen2.5-coder:7b",
    top_k: int = 4,
) -> dict[str, Any]:
    search = semantic_search(query, collection_name=collection_name, embed_model=embed_model, top_k=top_k)
    if not search.get("ok"):
        return search

    results = search["results"]
    context_blocks = []
    for i, row in enumerate(results, start=1):
        context_blocks.append(f"[{i}] Fuente: {row['source']}\n{row['text']}")

    context_text = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "Eres NousGPT-RAG, tutor de DAM2 (Acceso a datos). "
                "Responde solo con información sustentada en el contexto recuperado. "
                "Si falta información, dilo claramente. "
                "Termina con un bloque 'Fuentes usadas' citando [n] y nombre de fuente."
            ),
        },
        {
            "role": "user",
            "content": f"Consulta del alumno:\n{query}\n\nContexto recuperado:\n{context_text}",
        },
    ]

    answer = _ollama_chat(messages, model=chat_model)

    return {
        "ok": True,
        "query": query,
        "answer": answer,
        "results": results,
        "chat_model": chat_model,
        "embed_model": embed_model,
    }


def collection_status(collection_name: str = DEFAULT_COLLECTION) -> dict[str, Any]:
    collection = get_or_create_collection(collection_name)
    return {
        "ok": True,
        "collection": collection_name,
        "chunks": collection.count(),
    }
