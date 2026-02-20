# NousGPT-RAG — Plantilla de Examen

**Alumno:** Luis Rodríguez Cedeño · **DNI:** 53945291X  
**Módulo:** Acceso a Datos · **Curso:** DAM2 2025/26

---

## 1. Introducción

- **Qué es:** Sistema RAG (Retrieval-Augmented Generation) con ChromaDB + Ollama
- **Contexto:** Módulo de Acceso a Datos — ingestión de corpus, embeddings vectoriales, búsqueda semántica, generación aumentada
- **Objetivos principales:**
  - Ingestión de corpus (archivos .md y .txt) con chunking (ventana deslizante con overlap)
  - Generación de embeddings con Ollama (`/api/embeddings`)
  - Almacenamiento vectorial en ChromaDB (coseno)
  - Búsqueda semántica y generación de respuestas con contexto relevante
  - API REST Flask con frontend integrado
- **Tecnologías clave:**
  - Python 3.11, Flask, ChromaDB (base de datos vectorial), Ollama (embeddings + chat)
  - Chunking con ventana deslizante, distancia coseno, prompt engineering
- **Arquitectura:** `app.py` (Flask endpoints) → `rag_engine.py` (motor RAG: chunk, embed, search, answer) → `config.py` (constantes) → `corpus/` (documentos fuente .md/.txt)

---

## 2. Desarrollo de las partes

### 2.1 Chunking — Ventana deslizante con overlap

- Divide textos largos en trozos de tamaño fijo (CHUNK_SIZE) con solapamiento (CHUNK_OVERLAP)
- Asegura que el contexto no se pierda en los bordes entre chunks

```python
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Dividir texto en chunks con ventana deslizante solapada."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap  # avance = size - overlap (solapa)
    return chunks
```

> **Explicación:** Se trocea el texto en bloques de `size` palabras, avanzando `size - overlap` cada vez. Esto garantiza que cada chunk comparte `overlap` palabras con el anterior, evitando cortar ideas a mitad.

### 2.2 Generación de embeddings con Ollama

- POST a `/api/embeddings` de Ollama con el modelo configurado
- Cada chunk se convierte en un vector numérico (embedding)
- Se usa el modelo `nomic-embed-text` por defecto

```python
def _ollama_embeddings(texts: list[str]) -> list[list[float]]:
    """Generar embeddings llamando a Ollama /api/embeddings."""
    results = []
    for text in texts:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120
        )
        resp.raise_for_status()
        results.append(resp.json()["embedding"])
    return results
```

> **Explicación:** Se envía cada texto al endpoint de embeddings de Ollama. Devuelve un vector de números (dimensiones del embedding). Se usa el modelo `nomic-embed-text` configurado en `config.py`.

### 2.3 ChromaDB — Almacenamiento vectorial

- `chromadb.PersistentClient` → almacena embeddings persistentemente en disco
- `get_or_create_collection()` → colección con distancia coseno
- `upsert()` para insertar/actualizar chunks

```python
import chromadb

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_collection(name: str = COLLECTION_NAME):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # distancia coseno
    )

def train_collection(corpus_dir: str = CORPUS_DIR):
    """Ingestión: leer corpus → chunk → embed → almacenar en ChromaDB."""
    collection = get_collection()
    for file in Path(corpus_dir).glob("*"):
        if file.suffix in (".md", ".txt"):
            text = file.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            embeddings = _ollama_embeddings(chunks)
            ids = [f"{file.stem}_{i}" for i in range(len(chunks))]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": file.name}] * len(chunks)
            )
```

> **Explicación:** `PersistentClient` almacena en disco (no se pierden al reiniciar). La colección usa distancia coseno para medir similitud. `train_collection()` lee todos los archivos del corpus, los chunquea, genera embeddings y los inserta en ChromaDB.

### 2.4 Búsqueda semántica

- Convierte la query del usuario en embedding
- Busca los N chunks más similares en ChromaDB por distancia coseno
- Devuelve documentos con score de similitud

```python
def semantic_search(query: str, n_results: int = TOP_K) -> list[dict]:
    """Buscar chunks similares a la query en ChromaDB."""
    collection = get_collection()
    query_embedding = _ollama_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "distance": results["distances"][0][i],
            "source": results["metadatas"][0][i].get("source", "")
        })
    return output
```

> **Explicación:** Se embede la consulta del usuario, se buscan los `n_results` chunks más cercanos por coseno en ChromaDB. Devuelve texto, distancia y fuente de cada resultado.

### 2.5 Generación de respuesta (RAG answer)

- Construye un prompt con bloques de contexto (chunks recuperados)
- System prompt: instrucciones de rol + reglas de citación
- Llama a Ollama `/api/chat` con historial + contexto

```python
def rag_answer(question: str, history: list = None) -> dict:
    """Buscar contexto + generar respuesta con Ollama."""
    results = semantic_search(question)

    context_blocks = "\n\n".join(
        f"[Fragmento {i+1} — {r['source']}]\n{r['text']}"
        for i, r in enumerate(results)
    )

    system = (
        "Eres un asistente académico. Responde usando SOLO la información "
        "proporcionada en los fragmentos de contexto. Si no tienes información "
        "suficiente, indícalo claramente. Cita las fuentes."
    )

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Contexto:\n{context_blocks}\n\nPregunta: {question}"})

    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False}
    )
    reply = resp.json()["message"]["content"]

    return {"answer": reply, "sources": results}
```

> **Explicación:** Primero busca chunks relevantes. Luego construye un prompt que incluye los fragmentos como contexto. El system prompt obliga al modelo a usar solo la información dada. La respuesta incluye las fuentes utilizadas.

---

## 3. Presentación del proyecto

- **Flujo RAG completo:** Corpus → Chunking → Embeddings → ChromaDB → Query → Búsqueda semántica → Contexto → Generación
- **Demo:** `python app.py` → entrenar corpus → hacer preguntas → ver fuentes citadas
- **Punto fuerte:** Pipeline RAG end-to-end con Ollama local (sin APIs externas de pago)
- **Configuración fácil:** `config.py` con todas las constantes (modelo, chunk size, top_k)

---

## 4. Conclusión

- **Competencias:** Embeddings vectoriales, ChromaDB, chunking, prompt engineering, búsqueda semántica
- **Concepto RAG:** Retrieval-Augmented Generation = buscar contexto relevante antes de generar
- **Ventaja Ollama local:** Sin coste, privacidad de datos, modelo personalizable
- **Extensibilidad:** Añadir más formatos de corpus, cambiar modelo de embedding, ajustar chunk size
- **Valoración:** Demuestra técnica moderna de IA (RAG) con acceso a datos vectoriales
