# ChromaDB y RAG en local

ChromaDB almacena embeddings para búsqueda por similitud semántica.

Flujo habitual:
1. Cargar corpus documental.
2. Dividir en chunks.
3. Generar embeddings con modelo como nomic-embed-text en Ollama.
4. Guardar chunks + metadatos + embeddings en una colección.
5. Recuperar top-k chunks relevantes ante una pregunta.
6. Construir prompt con contexto recuperado y consultar un modelo de chat.

Este enfoque permite respuestas más precisas y con base documental.
