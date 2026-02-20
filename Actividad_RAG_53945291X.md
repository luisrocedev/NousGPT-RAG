# Actividad 004 - Entrenamiento de IA semantica (ChromaDB + RAG)

**DNI:** 53945291X
**Curso:** DAM2 - Acceso a datos
**Leccion:** `dam2526/Segundo/Acceso a datos/301-Actividades final de unidad - Segundo trimestre/004- Entrenamiento de una inteligencia artificial semantica - ChromaDB - RAG`

---

## 1) Proyecto desarrollado

He desarrollado el software **NousGPT RAG Lab**, un prototipo funcional de busqueda semantica y respuesta contextual (RAG - Retrieval Augmented Generation) sobre un corpus documental propio.

Ruta del proyecto:

- `004- Entrenamiento de una inteligencia artificial semantica - ChromaDB - RAG/rag_nousgpt/`

---

## 2) Pilar visual: modificaciones esteticas y UX

Partiendo del workflow tecnico de clase (Ollama + Chroma), se ha construido una interfaz web mas completa y clara para uso docente:

- Panel visual dividido en tres bloques: entrenamiento, busqueda semantica y pregunta RAG.
- Estilo minimalista profesional, con tarjetas, feedback de estado y resultados legibles.
- Renderizado de resultados por ranking y score de similitud con barra visual de score.
- Flujo de uso guiado para entrenar y consultar sin tocar codigo.
- Modo oscuro con persistencia (toggle manual + deteccion automatica del sistema).
- Badge de estado de Ollama en tiempo real al cargar la pagina.
- Barra informativa de corpus (ficheros disponibles).
- Renderizado basico de Markdown en las respuestas RAG (negritas, cursivas, bloques de codigo).
- Separacion visual de contenido de respuesta y fuentes utilizadas.
- Footer con creditos del proyecto.

Estas mejoras hacen que el prototipo sea demostrable en clase de forma directa, sin depender de scripts sueltos por terminal.

---

## 3) Pilar funcional: modificaciones de mucho calado

### 3.1 Motor de entrenamiento vectorial

- Carga automatica de corpus desde carpeta (`.md`, `.txt`, `.html`).
- Normalizacion de contenido y chunking configurable (`chunk_size`, `overlap`).
- Generacion de embeddings con `nomic-embed-text` via API local de Ollama.
- Persistencia de embeddings y metadatos en coleccion ChromaDB (`PersistentClient`).
- Opcion de resetear coleccion antes de reentrenar.

### 3.2 Busqueda semantica real

- Query embedding de la consulta del usuario con el mismo modelo de embeddings.
- Recuperacion Top-K en ChromaDB con distancia coseno.
- Devolucion de resultados con score, fuente, texto del chunk y barra visual de relevancia.

### 3.3 Capa RAG completa

- Construccion automatica de contexto con los chunks recuperados.
- Prompting con reglas para responder solo con evidencia del corpus.
- Generacion de respuesta con modelo de chat configurable en Ollama (por defecto `qwen2.5-coder:7b`).
- Inclusion de fuentes usadas en la salida final, separadas visualmente del contenido.
- Modelo de chat centralizado en `config.py` (constante `CHAT_MODEL`), no hardcodeado.

### 3.4 Backend API para explotacion

Se implementan endpoints dedicados:

- `POST /api/train` -> entrenamiento de coleccion vectorial.
- `POST /api/search` -> busqueda semantica por query.
- `POST /api/ask` -> respuesta RAG con contexto recuperado.
- `GET /api/status` -> estado de coleccion (nombre, numero de chunks).
- `GET /api/ollama` -> verificacion de disponibilidad de Ollama y listado de modelos instalados.
- `GET /api/corpus` -> listado de ficheros del corpus con nombre, extension y tamano.

Esta separacion permite reutilizacion futura por otras interfaces o servicios.

### 3.5 Diagnostico y verificacion

- Funcion `check_ollama()` integrada en el motor: verifica que el servidor Ollama esta activo y lista los modelos instalados.
- Funcion `list_corpus_files()`: devuelve los ficheros del directorio corpus con metadatos (extension, tamano en bytes).
- Ambas funciones expuestas como endpoints API para consumo desde el frontend.

---

## 4) Arquitectura tecnica

### 4.1 Configuracion centralizada (`config.py`)

Todas las constantes del sistema se gestionan desde un unico modulo:

```python
OLLAMA_BASE_URL    = "http://localhost:11434"
EMBED_MODEL        = "nomic-embed-text"
CHAT_MODEL         = "qwen2.5-coder:7b"
DEFAULT_COLLECTION = "nousgpt_rag"
DEFAULT_TOP_K      = 4
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120
CORPUS_DIR         = os.path.join(os.path.dirname(__file__), "corpus")
DATA_DIR           = os.path.join(os.path.dirname(__file__), "data")
```

Esto permite cambiar modelo, tamano de chunk o ruta de corpus sin tocar el motor.

### 4.2 Motor RAG (`rag_engine.py`)

Funciones principales:

| Funcion | Descripcion |
|---|---|
| `chunk_text(text, size, overlap)` | Divide texto en fragmentos solapados |
| `load_corpus_chunks(corpus_dir, ...)` | Carga y fragmenta todos los ficheros del corpus |
| `_ollama_embeddings(texts, model)` | Genera embeddings via API de Ollama |
| `_ollama_chat(prompt, model)` | Genera respuesta de chat via API de Ollama |
| `train_collection(...)` | Entrena coleccion ChromaDB con corpus completo |
| `semantic_search(query, ...)` | Busca los chunks mas similares a una consulta |
| `rag_answer(query, ...)` | Busca contexto + genera respuesta RAG |
| `collection_status(collection)` | Devuelve estado de la coleccion |
| `check_ollama()` | Verifica disponibilidad de Ollama y lista modelos |
| `list_corpus_files()` | Lista ficheros del corpus con metadatos |

### 4.3 Servidor Flask (`app.py`)

- Sirve la SPA desde `templates/index.html`.
- Expone API REST JSON en `/api/*`.
- Manejo de errores con try/except en cada endpoint.
- Puerto configurable (por defecto 5050).

### 4.4 Frontend (`templates/index.html` + `static/`)

- HTML semantico con estructura panel -> grid -> cards.
- CSS con custom properties, dark mode, animaciones `fadeIn`/`slideUp`, responsive.
- JavaScript vanilla (ES modules), sin dependencias externas.
- Renderizado markdown basico para respuestas.
- Persistencia de preferencia de dark mode en `localStorage`.

---

## 5) Base de datos semantica y corpus

- Motor vectorial: **ChromaDB PersistentClient** (persistencia en disco en `data/`).
- Coleccion por defecto: `nousgpt_rag`.
- Corpus inicial incluido con contenido de Acceso a datos:
  - transacciones ACID,
  - conexion segura con PDO,
  - flujo tecnico Chroma + RAG.
- Soporte para ficheros `.md`, `.txt` y `.html` en la carpeta `corpus/`.

---

## 6) Flujo de funcionamiento

1. El usuario abre la interfaz web (`http://localhost:5050`).
2. Se verifican automaticamente: estado de la coleccion, disponibilidad de Ollama y ficheros del corpus.
3. Panel **Entrenar**: el usuario configura chunk size, overlap, modelo de embedding y pulsa "Entrenar".
4. El motor carga los ficheros del corpus, los fragmenta, genera embeddings y los almacena en ChromaDB.
5. Panel **Buscar**: el usuario escribe una consulta y obtiene los chunks mas similares con score y fuente.
6. Panel **Preguntar RAG**: el usuario formula una pregunta, el sistema recupera contexto y genera respuesta con el LLM.
7. La respuesta se muestra con formato Markdown basico y las fuentes utilizadas aparecen separadas.

---

## 7) Relacion con el workflow de clase

Se mantiene el enfoque trabajado en clase:

- Uso de **Ollama** para embeddings semanticos y generacion de texto.
- Almacenamiento vectorial en **ChromaDB**.
- Recuperacion por similitud coseno en lugar de busqueda literal.

A partir de ese flujo base, se amplia a un software mas completo con:

- Interfaz web profesional con dark mode.
- API modular con 6 endpoints.
- Fase RAG integrada con renderizado markdown.
- Diagnostico de Ollama y corpus desde el propio frontend.

---

## 8) Archivos principales entregados

| Archivo | Descripcion |
|---|---|
| `app.py` | Servidor Flask con API REST |
| `rag_engine.py` | Motor RAG: chunking, embeddings, busqueda, respuesta |
| `config.py` | Configuracion centralizada |
| `templates/index.html` | SPA con dark mode y paneles |
| `static/styles.css` | Estilos CSS con custom properties y responsive |
| `static/app.js` | Logica del cliente: API, dark mode, markdown |
| `corpus/*.md` | Documentos fuente para entrenamiento |
| `requirements.txt` | Dependencias Python |
| `README.md` | Documentacion del proyecto |

---

## 9) Pruebas funcionales realizadas

| Prueba | Estado |
|---|---|
| Entrenamiento de coleccion con corpus local | PASS |
| Persistencia de chunks vectoriales en Chroma | PASS |
| Busqueda semantica Top-K con score | PASS |
| Respuesta RAG con contexto recuperado | PASS |
| Visualizacion completa desde interfaz web | PASS |
| Dark mode toggle + persistencia localStorage | PASS |
| Badge Ollama online/offline | PASS |
| Listado de ficheros del corpus | PASS |
| Renderizado Markdown basico en respuestas | PASS |
| Separacion visual de respuesta y fuentes | PASS |

---

## 10) Conclusion

La actividad se entrega como un prototipo funcional y extensible que demuestra dominio practico de:

- bases de datos vectoriales (ChromaDB),
- embeddings semanticos (nomic-embed-text via Ollama),
- recuperacion de contexto por similitud coseno,
- generacion aumentada por recuperacion (RAG),
- API REST con Flask,
- interfaz web moderna con dark mode y Markdown rendering.

El proyecto esta publicado en GitHub como repositorio independiente.

**Estado final:** Completado y listo para evaluacion.
