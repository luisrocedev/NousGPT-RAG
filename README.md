# NousGPT RAG Lab

Prototipo funcional de entrenamiento semántico con **ChromaDB + Ollama** para DAM2 (Acceso a datos).

## Qué hace

- Entrena una colección vectorial a partir de un corpus (`.md`, `.txt`, `.html`).
- Genera embeddings locales con Ollama (`nomic-embed-text`).
- Almacena chunks y metadatos en ChromaDB persistente.
- Permite búsqueda semántica Top-K.
- Genera respuestas RAG con fuentes usando un modelo de chat Ollama.

## Stack

- Python 3.11+
- Flask
- ChromaDB (PersistentClient)
- Ollama (`/api/embeddings` y `/api/chat`)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requisitos Ollama

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

## Ejecutar

```bash
python app.py
```

Abrir: `http://127.0.0.1:5050`

## Flujo de uso

1. Revisar ruta de corpus (por defecto `./corpus`).
2. Pulsar **Entrenar base vectorial**.
3. Probar **Búsqueda semántica** con una consulta.
4. Lanzar **Pregunta RAG** para respuesta contextual con fuentes.

## Estructura

- `app.py`: servidor Flask y endpoints API.
- `rag_engine.py`: chunking, embeddings, entrenamiento, búsqueda y RAG.
- `config.py`: configuración central.
- `corpus/`: documentos base de entrenamiento.
- `static/` + `templates/`: interfaz web.
