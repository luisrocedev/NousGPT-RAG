<div align="center">

# NousGPT RAG Lab

**Motor de busqueda semantica y respuesta contextual con ChromaDB + Ollama**

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000?logo=flask)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-ff6f00)
![Ollama](https://img.shields.io/badge/Ollama-local_LLM-1a1a2e)
![License](https://img.shields.io/badge/License-MIT-22c55e)

Entrena un corpus documental, busca por similitud semantica y genera respuestas RAG con fuentes -- todo desde una interfaz web moderna con dark mode.

</div>

---

## Caracteristicas

| Modulo                      | Descripcion                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------ |
| **Entrenamiento vectorial** | Carga `.md` / `.txt` / `.html`, fragmenta con overlap y genera embeddings con Ollama |
| **ChromaDB persistente**    | Almacena chunks y metadatos en disco (`PersistentClient`)                            |
| **Busqueda semantica**      | Recupera Top-K chunks por distancia coseno con score y fuente                        |
| **Respuesta RAG**           | Construye contexto, prompting controlado y genera respuesta con LLM local            |
| **API REST**                | 6 endpoints JSON para entrenar, buscar, preguntar, diagnostico                       |
| **Dark mode**               | Toggle manual + deteccion automatica del sistema, persistencia en localStorage       |
| **Ollama status**           | Verificacion en tiempo real del servidor Ollama y modelos instalados                 |
| **Corpus info**             | Listado de ficheros del corpus con extension y tamano                                |
| **Markdown render**         | Respuestas RAG con negritas, cursivas, bloques de codigo                             |

---

## Arquitectura

```
config.py          Constantes centralizadas
    |
rag_engine.py      Motor: chunk -> embed -> store -> search -> RAG
    |
app.py             Flask API  (6 endpoints)
    |
templates/ + static/   SPA  (HTML + CSS + JS vanilla)
    |
corpus/            Documentos fuente (.md, .txt, .html)
    |
data/              ChromaDB  (persistencia en disco)
```

---

## Instalacion

```bash
# 1. Clonar
git clone https://github.com/luisrocedev/NousGPT-RAG.git
cd NousGPT-RAG

# 2. Entorno virtual
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ollama (debe estar corriendo)
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

---

## Uso

```bash
python app.py
```

Abrir `http://127.0.0.1:5050`

### Flujo

1. **Entrenar** -- configurar chunk size, overlap, modelo de embedding y pulsar "Entrenar base vectorial".
2. **Buscar** -- escribir una consulta y obtener los chunks mas similares con score.
3. **Preguntar RAG** -- formular una pregunta, el sistema recupera contexto y genera respuesta con fuentes.

---

## API Endpoints

| Metodo | Ruta          | Descripcion                                   |
| ------ | ------------- | --------------------------------------------- |
| `GET`  | `/api/status` | Estado de la coleccion (nombre, chunks)       |
| `GET`  | `/api/ollama` | Disponibilidad de Ollama y modelos instalados |
| `GET`  | `/api/corpus` | Listado de ficheros del corpus                |
| `POST` | `/api/train`  | Entrenar coleccion vectorial                  |
| `POST` | `/api/search` | Busqueda semantica por query                  |
| `POST` | `/api/ask`    | Respuesta RAG con contexto recuperado         |

---

## Stack

- **Python 3.11+** -- lenguaje principal
- **Flask 3.0** -- servidor web y API REST
- **ChromaDB 0.5** -- base de datos vectorial con persistencia
- **Ollama** -- embeddings (`nomic-embed-text`) y chat (`qwen2.5-coder:7b`)
- **HTML + CSS + JS** -- interfaz SPA con dark mode y markdown rendering

---

## Estructura del proyecto

```
NousGPT-RAG/
  app.py              Servidor Flask
  rag_engine.py        Motor RAG completo
  config.py            Configuracion centralizada
  requirements.txt     Dependencias Python
  corpus/              Documentos fuente
    *.md
  data/                ChromaDB (auto-generado)
  templates/
    index.html         SPA principal
  static/
    app.js             Logica del cliente
    styles.css          Estilos con dark mode
```

---

## Configuracion

Editar `config.py` para ajustar:

| Variable                | Default                  | Descripcion                       |
| ----------------------- | ------------------------ | --------------------------------- |
| `OLLAMA_BASE_URL`       | `http://localhost:11434` | URL del servidor Ollama           |
| `EMBED_MODEL`           | `nomic-embed-text`       | Modelo de embeddings              |
| `CHAT_MODEL`            | `qwen2.5-coder:7b`       | Modelo de chat LLM                |
| `DEFAULT_COLLECTION`    | `nousgpt_rag`            | Nombre de la coleccion ChromaDB   |
| `DEFAULT_TOP_K`         | `4`                      | Numero de resultados por busqueda |
| `DEFAULT_CHUNK_SIZE`    | `700`                    | Tamano de fragmento (caracteres)  |
| `DEFAULT_CHUNK_OVERLAP` | `120`                    | Solapamiento entre fragmentos     |

---

## Autor

**Luis Rodriguez** -- [github.com/luisrocedev](https://github.com/luisrocedev)

---

<div align="center">
<sub>NousGPT RAG Lab -- Retrieval Augmented Generation con ChromaDB + Ollama</sub>
</div>
