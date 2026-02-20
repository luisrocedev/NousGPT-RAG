# Sistema RAG con ChromaDB y Ollama - NousGPT RAG Lab

**DNI:** 53945291X  
**Curso:** DAM2 — Acceso a datos  
**Actividad:** 004-Entrenamiento de IA semántica con ChromaDB y RAG  
**Tecnologías:** Python 3.13 · ChromaDB · Ollama · Flask · JavaScript  
**Fecha:** 10 de febrero de 2026

---

## 1. Introducción breve y contextualización (25%)

### Concepto general

RAG (Retrieval-Augmented Generation) es una técnica que combina búsqueda semántica en bases de datos vectoriales con generación de texto mediante IA. En lugar de que el modelo genere respuestas solo desde su conocimiento general, primero recupera información relevante de un corpus de documentos específico y luego genera respuestas basadas en ese contexto recuperado.

Un sistema RAG consta de tres componentes principales:

1. **Base de datos vectorial:** Almacena representaciones numéricas (embeddings) de documentos
2. **Motor de búsqueda semántica:** Encuentra documentos similares a una consulta
3. **Generador de respuestas:** Crea respuestas coherentes usando el contexto recuperado

### Contexto y utilidad

Los sistemas RAG son fundamentales en aplicaciones empresariales y académicas porque:

- **Conocimiento actualizable:** Permiten añadir información nueva sin reentrenar el modelo de IA
- **Respuestas verificables:** Las respuestas están basadas en documentos específicos que pueden citarse
- **Dominio específico:** Funcionan con conocimiento especializado que los modelos generales no tienen
- **Privacidad:** Los datos permanecen en local sin enviarlos a servicios externos

Este proyecto implementa un sistema RAG completo que permite entrenar una base de datos semántica con documentos propios y hacer consultas sobre ese conocimiento específico.

---

## 2. Desarrollo detallado y preciso (25%)

### Arquitectura del sistema RAG

**Proceso de entrenamiento (indexación):**

1. Leer documentos del corpus
2. Dividir en fragmentos (chunks) manejables
3. Generar embeddings vectoriales de cada chunk
4. Almacenar en ChromaDB con metadatos

**Proceso de consulta (retrieval):**

1. Convertir pregunta del usuario en embedding
2. Buscar chunks más similares en ChromaDB
3. Construir contexto con los chunks recuperados
4. Generar respuesta usando el modelo de IA con ese contexto

### Base de datos vectorial con ChromaDB

ChromaDB es una base de datos especializada en embeddings que permite búsqueda por similitud:

```python
# rag_engine.py
import chromadb
from chromadb.config import Settings
import requests
import json
from pathlib import Path
from typing import List, Dict, Any

class RAGEngine:
    def __init__(self, collection_name: str = "nousgpt_rag"):
        # Configurar cliente persistente de ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Crear o recuperar colección
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Colección '{collection_name}' cargada")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Corpus académico RAG"}
            )
            print(f"Colección '{collection_name}' creada")

        self.ollama_url = "http://localhost:11434"
        self.embed_model = "nomic-embed-text"
        self.chat_model = "qwen2.5-coder:7b"
```

### Generación de embeddings con Ollama

Los embeddings son representaciones vectoriales que capturan el significado semántico:

```python
def generate_embedding(self, text: str) -> List[float]:
    """Genera embedding vectorial para un texto usando Ollama"""
    url = f"{self.ollama_url}/api/embeddings"

    payload = {
        "model": self.embed_model,
        "prompt": text
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result['embedding']
    else:
        raise Exception(f"Error al generar embedding: {response.status_code}")
```

### Sistema de chunking de documentos

Dividir documentos en fragmentos optimiza la recuperación:

```python
def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide texto en chunks con solapamiento"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks

def load_corpus(self, corpus_path: str) -> List[Dict[str, Any]]:
    """Carga documentos desde una carpeta"""
    documents = []
    corpus_dir = Path(corpus_path)

    # Formatos soportados
    extensions = ['.txt', '.md', '.html']

    for file_path in corpus_dir.rglob('*'):
        if file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Dividir en chunks
                chunks = self.chunk_text(content)

                for idx, chunk in enumerate(chunks):
                    documents.append({
                        'text': chunk,
                        'source': file_path.name,
                        'chunk_id': f"{file_path.stem}_{idx}"
                    })
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")

    return documents
```

### Entrenamiento de la colección

```python
def train(self, corpus_path: str) -> Dict[str, Any]:
    """Entrena la colección con documentos del corpus"""
    print(f"Cargando corpus desde {corpus_path}...")
    documents = self.load_corpus(corpus_path)

    if not documents:
        return {'success': False, 'error': 'No se encontraron documentos'}

    print(f"Documentos cargados: {len(documents)}")
    print("Generando embeddings...")

    ids = []
    embeddings = []
    metadatas = []
    documents_text = []

    for doc in documents:
        # Generar embedding
        embedding = self.generate_embedding(doc['text'])

        ids.append(doc['chunk_id'])
        embeddings.append(embedding)
        documents_text.append(doc['text'])
        metadatas.append({
            'source': doc['source'],
            'chunk_id': doc['chunk_id']
        })

    # Insertar en ChromaDB
    print("Insertando en ChromaDB...")
    self.collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_text,
        metadatas=metadatas
    )

    return {
        'success': True,
        'documents_processed': len(documents),
        'collection_count': self.collection.count()
    }
```

### Búsqueda semántica

```python
def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Búsqueda semántica en la colección"""
    # Generar embedding de la consulta
    query_embedding = self.generate_embedding(query)

    # Buscar en ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    # Formatear resultados
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
            'similarity_score': 1 - results['distances'][0][i]  # Convertir distancia a similitud
        })

    return formatted_results
```

### Sistema RAG completo

```python
def ask(self, question: str, n_context: int = 3) -> Dict[str, Any]:
    """Genera respuesta RAG basada en contexto recuperado"""
    # 1. Búsqueda semántica
    search_results = self.search(question, n_results=n_context)

    if not search_results:
        return {
            'answer': 'No se encontró información relevante en el corpus.',
            'sources': []
        }

    # 2. Construir contexto
    context = "\n\n".join([
        f"[Fuente: {r['metadata']['source']}]\n{r['document']}"
        for r in search_results
    ])

    # 3. Crear prompt con instrucciones
    system_prompt = """Eres un asistente académico. Responde SOLO basándote en el contexto proporcionado.
Si la información no está en el contexto, indícalo claramente.
Cita las fuentes cuando sea relevante."""

    user_prompt = f"""Contexto:
{context}

Pregunta: {question}

Respuesta:"""

    # 4. Generar respuesta con Ollama
    url = f"{self.ollama_url}/api/chat"
    payload = {
        "model": self.chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        answer = response.json()['message']['content']
        sources = list(set([r['metadata']['source'] for r in search_results]))

        return {
            'answer': answer,
            'sources': sources,
            'context_chunks': len(search_results)
        }
    else:
        raise Exception(f"Error al generar respuesta: {response.status_code}")
```

### API REST con Flask

```python
# app.py
from flask import Flask, request, jsonify, render_template
from rag_engine import RAGEngine

app = Flask(__name__)
rag = RAGEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    """Endpoint para entrenar la colección"""
    data = request.json
    corpus_path = data.get('corpus_path', './corpus')

    try:
        result = rag.train(corpus_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Endpoint para búsqueda semántica"""
    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 5)

    try:
        results = rag.search(query, n_results)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    """Endpoint para pregunta RAG"""
    data = request.json
    question = data.get('question', '')
    n_context = data.get('n_context', 3)

    try:
        response = rag.ask(question, n_context)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Estado de la colección"""
    try:
        count = rag.collection.count()
        return jsonify({
            'collection_name': rag.collection.name,
            'document_count': count,
            'status': 'ready' if count > 0 else 'empty'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Terminología técnica

- **Embedding:** Vector numérico de alta dimensionalidad que representa el significado semántico de un texto
- **Similitud coseno:** Medida de similitud entre vectores basada en el ángulo entre ellos
- **Chunking:** División de documentos largos en fragmentos manejables
- **Overlap:** Solapamiento entre chunks consecutivos para mantener contexto
- **RAG (Retrieval-Augmented Generation):** Técnica que combina búsqueda de información con generación de texto

---

## 3. Aplicación práctica (25%)

### Ejemplo completo de uso

```python
# ejemplo_uso.py
from rag_engine import RAGEngine

# Inicializar motor RAG
rag = RAGEngine(collection_name="biblioteca_academica")

# 1. Entrenar con corpus de documentos
print("=== ENTRENAMIENTO ===")
resultado = rag.train('./corpus')
print(f"Documentos procesados: {resultado['documents_processed']}")
print(f"Total en colección: {resultado['collection_count']}")

# 2. Búsqueda semántica
print("\n=== BÚSQUEDA SEMÁNTICA ===")
query = "¿Qué son las transacciones ACID?"
resultados = rag.search(query, n_results=3)

for i, res in enumerate(resultados, 1):
    print(f"\nResultado {i}:")
    print(f"Similitud: {res['similarity_score']:.3f}")
    print(f"Fuente: {res['metadata']['source']}")
    print(f"Texto: {res['document'][:200]}...")

# 3. Pregunta RAG con contexto
print("\n=== RESPUESTA RAG ===")
pregunta = "Explica qué es una transacción ACID y sus propiedades"
respuesta = rag.ask(pregunta, n_context=3)

print(f"Pregunta: {pregunta}")
print(f"\nRespuesta:\n{respuesta['answer']}")
print(f"\nFuentes consultadas: {', '.join(respuesta['sources'])}")
```

### Cliente JavaScript

```javascript
// static/app.js
class RAGClient {
  constructor() {
    this.setupEventListeners();
    this.loadStatus();
  }

  setupEventListeners() {
    document.getElementById("btn-train").addEventListener("click", () => {
      this.train();
    });

    document.getElementById("btn-search").addEventListener("click", () => {
      this.search();
    });

    document.getElementById("btn-ask").addEventListener("click", () => {
      this.ask();
    });
  }

  async loadStatus() {
    try {
      const response = await fetch("/api/status");
      const data = await response.json();

      document.getElementById("status-info").innerHTML = `
                Colección: ${data.collection_name}<br>
                Documentos: ${data.document_count}<br>
                Estado: ${data.status}
            `;
    } catch (error) {
      console.error("Error al cargar estado:", error);
    }
  }

  async train() {
    const corpusPath = document.getElementById("corpus-path").value;
    const statusDiv = document.getElementById("train-status");

    statusDiv.textContent = "Entrenando...";

    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corpus_path: corpusPath }),
      });

      const result = await response.json();

      if (result.success) {
        statusDiv.innerHTML = `
                    ✅ Entrenamiento completado<br>
                    Documentos procesados: ${result.documents_processed}<br>
                    Total en colección: ${result.collection_count}
                `;
        this.loadStatus();
      } else {
        statusDiv.textContent = `❌ Error: ${result.error}`;
      }
    } catch (error) {
      statusDiv.textContent = `❌ Error: ${error.message}`;
    }
  }

  async search() {
    const query = document.getElementById("search-query").value;
    const nResults = document.getElementById("n-results").value;
    const resultsDiv = document.getElementById("search-results");

    resultsDiv.textContent = "Buscando...";

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          n_results: parseInt(nResults),
        }),
      });

      const data = await response.json();

      resultsDiv.innerHTML = data.results
        .map(
          (r, i) => `
                <div class="result-card">
                    <div class="result-header">
                        <strong>Resultado ${i + 1}</strong>
                        <span class="score">Similitud: ${(r.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-source">Fuente: ${r.metadata.source}</div>
                    <div class="result-text">${r.document}</div>
                </div>
            `,
        )
        .join("");
    } catch (error) {
      resultsDiv.textContent = `❌ Error: ${error.message}`;
    }
  }

  async ask() {
    const question = document.getElementById("rag-question").value;
    const nContext = document.getElementById("n-context").value;
    const answerDiv = document.getElementById("rag-answer");

    answerDiv.textContent = "Generando respuesta...";

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          n_context: parseInt(nContext),
        }),
      });

      const data = await response.json();

      answerDiv.innerHTML = `
                <div class="answer-card">
                    <h3>Respuesta:</h3>
                    <p>${data.answer}</p>
                    <div class="sources">
                        <strong>Fuentes consultadas:</strong> ${data.sources.join(", ")}
                    </div>
                    <div class="meta">
                        Chunks de contexto usados: ${data.context_chunks}
                    </div>
                </div>
            `;
    } catch (error) {
      answerDiv.textContent = `❌ Error: ${error.message}`;
    }
  }
}

// Inicializar
const ragClient = new RAGClient();
```

### Errores comunes y soluciones

**Error 1:** No verificar que Ollama esté ejecutándose.

```python
# Incorrecto - No maneja error de conexión
embedding = self.generate_embedding(text)

# Correcto - Verifica disponibilidad
def check_ollama_available(self):
    try:
        response = requests.get(f"{self.ollama_url}/api/tags")
        return response.status_code == 200
    except:
        return False

if not self.check_ollama_available():
    raise Exception("Ollama no está disponible en localhost:11434")
```

**Error 2:** No manejar documentos vacíos o muy cortos.

```python
# Incorrecto
chunks = self.chunk_text(content)

# Correcto
if len(content.strip()) < 10:
    print(f"Documento muy corto, omitiendo: {file_path}")
    continue

chunks = self.chunk_text(content)
chunks = [c for c in chunks if len(c.strip()) > 20]
```

**Error 3:** No normalizar distancias a scores de similitud.

```python
# Incorrecto - Distancia no es intuitiva
print(f"Distancia: {result['distances'][0][i]}")

# Correcto - Convertir a similitud (0-1)
similarity = 1 - result['distances'][0][i]
print(f"Similitud: {similarity:.3f}")
```

---

## 4. Conclusión breve (25%)

### Resumen de puntos clave

Este sistema RAG demuestra la integración de tecnologías modernas de IA:

1. **Base de datos vectorial:** ChromaDB para almacenamiento eficiente de embeddings
2. **Embeddings semánticos:** Generación con Ollama para capturar significado contextual
3. **Búsqueda por similitud:** Recuperación basada en distancia coseno en espacio vectorial
4. **Generación aumentada:** Respuestas de IA basadas en contexto recuperado del corpus

### Enlace con contenidos de la unidad

Este proyecto aplica conceptos avanzados del módulo:

- **Bases de datos especializadas (Unidad 4):** ChromaDB como base de datos vectorial
- **Componentes de acceso a datos (Unidad 6):** API REST para operaciones CRUD en colección vectorial
- **Integración de sistemas:** Combinación de ChromaDB (persistencia) con Ollama (IA)
- **Optimización de consultas:** Chunking y overlap para mejorar recuperación

La arquitectura RAG representa el estado del arte en sistemas de IA empresarial, permitiendo crear asistentes especializados con conocimiento actualizable sin reentrenamiento de modelos. La capacidad de citar fuentes y verificar respuestas hace que estos sistemas sean apropiados para entornos académicos y profesionales donde la precisión y trazabilidad son críticas.
