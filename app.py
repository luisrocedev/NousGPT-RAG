from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import (
    CHAT_MODEL,
    CORPUS_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_TOP_K,
    EMBED_MODEL,
)
from rag_engine import collection_status, rag_answer, semantic_search, train_collection

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_collection=DEFAULT_COLLECTION,
        default_top_k=DEFAULT_TOP_K,
        default_chunk_size=DEFAULT_CHUNK_SIZE,
        default_overlap=DEFAULT_CHUNK_OVERLAP,
        default_embed_model=EMBED_MODEL,
        default_chat_model=CHAT_MODEL,
        default_corpus=str(CORPUS_DIR),
    )


@app.get("/api/status")
def api_status():
    collection_name = request.args.get("collection", DEFAULT_COLLECTION)
    return jsonify(collection_status(collection_name=collection_name))


@app.post("/api/train")
def api_train():
    payload = request.get_json(silent=True) or {}

    collection_name = (payload.get("collection") or DEFAULT_COLLECTION).strip()
    embed_model = (payload.get("embed_model") or EMBED_MODEL).strip()
    corpus_dir = Path(payload.get("corpus_dir") or CORPUS_DIR)
    chunk_size = int(payload.get("chunk_size") or DEFAULT_CHUNK_SIZE)
    overlap = int(payload.get("overlap") or DEFAULT_CHUNK_OVERLAP)
    reset = bool(payload.get("reset", True))

    try:
        result = train_collection(
            corpus_dir=corpus_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            chunk_size=chunk_size,
            overlap=overlap,
            reset=reset,
        )
        code = 200 if result.get("ok") else 400
        return jsonify(result), code
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/search")
def api_search():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    collection_name = (payload.get("collection") or DEFAULT_COLLECTION).strip()
    embed_model = (payload.get("embed_model") or EMBED_MODEL).strip()
    top_k = int(payload.get("top_k") or DEFAULT_TOP_K)

    try:
        result = semantic_search(
            query,
            collection_name=collection_name,
            embed_model=embed_model,
            top_k=top_k,
        )
        code = 200 if result.get("ok") else 400
        return jsonify(result), code
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    collection_name = (payload.get("collection") or DEFAULT_COLLECTION).strip()
    embed_model = (payload.get("embed_model") or EMBED_MODEL).strip()
    chat_model = (payload.get("chat_model") or CHAT_MODEL).strip()
    top_k = int(payload.get("top_k") or DEFAULT_TOP_K)

    try:
        result = rag_answer(
            query,
            collection_name=collection_name,
            embed_model=embed_model,
            chat_model=chat_model,
            top_k=top_k,
        )
        code = 200 if result.get("ok") else 400
        return jsonify(result), code
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)
