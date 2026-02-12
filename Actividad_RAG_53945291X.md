# Actividad 004 · Entrenamiento de IA semántica (ChromaDB + RAG)

**DNI:** 53945291X  
**Curso:** DAM2 - Acceso a datos  
**Lección:** `dam2526/Segundo/Acceso a datos/301-Actividades final de unidad - Segundo trimestre/004- Entrenamiento de una inteligencia artificial semántica - ChromaDB - RAG`

---

## 1) Proyecto desarrollado

He desarrollado el software **NousGPT RAG Lab**, un prototipo funcional de búsqueda semántica y respuesta contextual (RAG) sobre un corpus documental propio.

Ruta del proyecto:

- `004- Entrenamiento de una inteligencia artificial semántica - ChromaDB - RAG/rag_nousgpt/`

---

## 2) Pilar visual: modificaciones estéticas y UX

Partiendo del workflow técnico de clase (Ollama + Chroma), se ha construido una interfaz web más completa y clara para uso docente:

- Panel visual dividido en tres bloques: entrenamiento, búsqueda semántica y pregunta RAG.
- Estilo minimalista profesional, con tarjetas, feedback de estado y resultados legibles.
- Renderizado de resultados por ranking y score de similitud.
- Flujo de uso guiado para entrenar y consultar sin tocar código.

Estas mejoras hacen que el prototipo sea demostrable en clase de forma directa, sin depender de scripts sueltos por terminal.

---

## 3) Pilar funcional: modificaciones de mucho calado

### 3.1 Motor de entrenamiento vectorial

- Carga automática de corpus desde carpeta (`.md`, `.txt`, `.html`).
- Normalización de contenido y chunking configurable (`chunk_size`, `overlap`).
- Generación de embeddings con `nomic-embed-text` vía API local de Ollama.
- Persistencia de embeddings y metadatos en colección ChromaDB.

### 3.2 Búsqueda semántica real

- Query embedding de la consulta del usuario.
- Recuperación Top-K en ChromaDB con distancia coseno.
- Devolución de resultados con score, fuente y texto del chunk.

### 3.3 Capa RAG completa

- Construcción automática de contexto con los chunks recuperados.
- Prompting con reglas para responder solo con evidencia del corpus.
- Generación de respuesta con modelo de chat en Ollama.
- Inclusión de fuentes usadas en la salida final.

### 3.4 Backend API para explotación

Se implementan endpoints dedicados:

- `POST /api/train` → entrenamiento de colección.
- `POST /api/search` → búsqueda semántica.
- `POST /api/ask` → respuesta RAG.
- `GET /api/status` → estado de colección.

Esta separación permite reutilización futura por otras interfaces o servicios.

---

## 4) Base de datos semántica y corpus

- Motor vectorial: **ChromaDB PersistentClient** (persistencia en disco).
- Colección por defecto: `nousgpt_rag`.
- Corpus inicial incluido con contenido de Acceso a datos:
  - transacciones ACID,
  - conexión segura con PDO,
  - flujo técnico Chroma + RAG.

---

## 5) Relación con el workflow de clase

Se mantiene el enfoque trabajado en clase:

- Uso de **Ollama** para embeddings semánticos.
- Almacenamiento en **ChromaDB**.
- Recuperación por similitud en lugar de búsqueda literal.

A partir de ese flujo base, se amplía a un software más completo con interfaz web, API modular y fase RAG integrada.

---

## 6) Archivos principales entregados

- `rag_nousgpt/app.py`
- `rag_nousgpt/rag_engine.py`
- `rag_nousgpt/config.py`
- `rag_nousgpt/templates/index.html`
- `rag_nousgpt/static/styles.css`
- `rag_nousgpt/static/app.js`
- `rag_nousgpt/corpus/*.md`
- `rag_nousgpt/README.md`

---

## 7) Pruebas funcionales realizadas

- Entrenamiento de colección con corpus local: ✅
- Persistencia de chunks vectoriales en Chroma: ✅
- Búsqueda semántica Top-K: ✅
- Respuesta RAG con contexto recuperado: ✅
- Visualización completa desde interfaz web: ✅

---

## 8) Conclusión

La actividad se entrega como un prototipo funcional y extensible que demuestra dominio práctico de:

- bases de datos vectoriales,
- embeddings semánticos,
- recuperación de contexto,
- y generación aumentada por recuperación (RAG).

**Estado final:** ✅ Completado y listo para evaluación.
