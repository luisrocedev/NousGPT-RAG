from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CORPUS_DIR = BASE_DIR / "corpus"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder:7b"

DEFAULT_COLLECTION = "nousgpt_rag"
DEFAULT_TOP_K = 4
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120
