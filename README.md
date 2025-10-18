# HRI RAG — Candidates Avanzados 2025

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** en Python  
para el reto de **Human-Robot Interaction (HRI)** de Roborregos Candidates 2025.

---

## Requisitos

- [Python](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [API key de OpenAI](https://platform.openai.com/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html)
## Librerias principales usadas

- **openai** - para embeddings y generación de respuestas con LLM.  
- **psycopg2-binary** - para conexión con Postgres + pgvector.  
- **python-dotenv** - para cargar variables de entorno desde `.env`.  
- **rank-bm25** - implementación de BM25 para recuperación léxica.  
- **re** (estándar) - para tokenización básica.
- **rclpy** — para la integración de ROS 2 y la comunicación entre nodos y servicios.  
- **docker / docker-compose** — para levantar los contenedores del sistema, base de datos y ROS 2.  
- **pgvector** — extensión de Postgres que permite almacenar y consultar embeddings vectoriales.  
- **functools** * — para implementar el sistema de caché con `lru_cache`.

---

## Cómo correrlo

1. **Clona este repo**
   ```bash
   git clone https://github.com/FabriBanda/HRI-ROBORREGOS.git
   cd HRI-ROBORREGOS
   ```

2. **Crea un entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Instala dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Crea un archivo `.env`**
   ```env
   OPENAI_API_KEY=sk-xxxx
   OPENAI_MODEL=gpt-4o-mini
   DB_URL=postgresql://postgres:postgres@localhost:5432/hri_rag
   ```

---

## Base de datos con Docker y pgvector

1. **Levantar Postgres con Docker** (requiere tener Docker instalado):
   ```bash
   docker compose up -d
   ```

---

## Ingestar la KB 

1. Coloca tus KB en la carpeta `kb/`.
2. Ejecuta el script:
   ```bash
   python ingest_pg.py
   ```

Esto:
- Lee los `.txt` de `kb/`.
- Genera embeddings con `text-embedding-3-small`.
- Inserta los chunks en la tabla `documents` de Postgres.

---

## Probar el sistema

Ejecuta el siguiente script:
 ```bash
python rag_pg.py
```

---

## Ejemplo de respuesta

```
RAG con Postgres+pgvector. Escribe 'quit' para salir.

Pregunta: ¿Cómo se llama el hermano de Fabricio?
El hermano de Fabricio es Rigoberto Banda-Hernández.

[Fuentes: kb_personal.txt#chunk4 (sim=0.78)]
```

---

---

## Add-ons implementados

### 1. Query Rewriting
Este paso adicional mejora la recuperación de contexto.  
Antes de buscar en la base vectorial, la pregunta del usuario se **reescribe automáticamente** con un LLM (chat gpt 4o mini) para hacerla más clara y concisa.

**Flujo:**
  1. Usuario hace una pregunta - `query`.
  2. El sistema la pasa por `rewrite_query()`.
  3. Se obtiene una versión más limpia y concisa - `q_eff`.
  4. La búsqueda en la DB vectorial se hace con `q_eff`.
  5. El LLM genera la respuesta usando el contexto recuperado.

## 2. Hybrid Retrieval (Embeddings + BM25)

Este sistema combina dos recuperadores:
- **Embeddings (pgvector)**: entienden el significado de la consulta (paráfrasis, sinonimos).
- **BM25 (léxico)**: se enfoca en coincidencias exactas (años, fechas, nombres).

## Recuperación semantica 
Cada fragmento de la KB se convierte en un vector usando el modelo text-embedding-3-small de OpenAI.
Estos vectores se guardan en Postgres con la extensión pgvecto y cada consulta, se genera un embedding de la pregunta y se buscan las partes más cercanas con distancia vectorial.

```
def retrieve(query: str, k: int = 3):
    qvec = embed(query)
    qvec_sql = f"[{', '.join(map(str, qvec))}]"
    sql = f"""
        SELECT chunk, source, chunk_idx, (embedding <-> '{qvec_sql}'::vector) AS l2
        FROM documents
        ORDER BY embedding <-> '{qvec_sql}'::vector
        LIMIT {int(k)}
    """
```

## Recuperación Lexica
Se construye un índice BM25 con todos los chunks almacenados en Postgres.
BM25 da prioridad a coincidencias exactas (años, nombres, números).

```
from rank_bm25 import BM25Okapi
def build_bm25_from_db():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk, source, chunk_idx FROM documents")
            rows = cur.fetchall()
    for chunk, source, idx in rows:
        BM25_DOCS.append(chunk)
        BM25_TOKENS.append(_tokenize(chunk))
        BM25_META.append((source, idx))
    BM25 = BM25Okapi(BM25_TOKENS)
```

## Fusion de Resultados

- Se toman los resultados de ambos métodos (semántico y léxico).
- Cada resultado se pondera según su posición en el ranking.
- La fórmula usada es 1 / (k + rank), con k=60 y al final, se devuelven los fragmentos con mayor puntaje combinado
  
```
def hybrid_retrieve(query: str, k: int = 4):
    sem = retrieve(query, k=k)
    lex = lexical_retrieve(query, k=k)

    sem_rank = { (d["source"], d["idx"]): r for r, d in enumerate(sem, start=1) }
    lex_rank = { (d["source"], d["idx"]): r for r, d in enumerate(lex, start=1) }

    fused = []
    for ky in set(sem_rank) | set(lex_rank):
        rrf = 0.0
        if ky in sem_rank: rrf += 1.0 / (60 + sem_rank[ky])
        if ky in lex_rank: rrf += 1.0 / (60 + lex_rank[ky])
        # ...
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    return fused[:k]
```
## 3. Text Caching 
Este sistema implementa **caching** en dos niveles para reducir latencia y costo de tokens.

### Caché de embeddings (`lru_cache`)
Cada vez que se genera un embedding para una consulta (`q_eff`), se guarda en memoria con `lru_cache`.  
Si la misma pregunta se repite el vector se devuelve al instante sin llamar a Chatgpt:

```
python
from functools import lru_cache

@lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple:
    return tuple(client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding)

def embed(text: str) -> list[float]:
    if ENABLE_CACHE:
        return list(_embed_cached(text))
    return client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding

```

### Caché de respuestas finales 
El sistema guarda la respuesta final del LLM por clave (q_eff + k)
Si la misma consulta se repite devuelve directamente la respuesta con el indicador [cache-hit] ( que define que esa pregunta fue hecha antes y por lo tanto no es necesario mandar llamar el LLM )

```
ANSWERS_CACHE: dict[str, str] = {}

def answer(query: str, k: int = 3) -> str:
    q_eff = rewrite_query(query) or query
    key = f"k={k}|{q_eff.strip().lower()}"
    if ENABLE_CACHE and key in ANSWERS_CACHE:
        return ANSWERS_CACHE[key] + "\n\n[cache-hit]"
    out = resp.choices[0].message.content
    if ENABLE_CACHE:
        ANSWERS_CACHE[key] = out
    return out

```

## 4. Task Distinction 
El sistema clasifica automaticamente cada pregunta en tres categorias:
### Skill
Preguntas asociadas a hora,fecha,mes,año,calculos y marcador de partidos
### LLM
Mensajes conversacionales simples que se contestan con el LLM
### KB (RAG)
Preguntas sobre las KB que de las cuales tiene conocimiento

## 5. Integración con ROS 2 

El sistema se integra con **ROS 2** para permitir la comunicación entre el modelo RAG y un entorno de interacción humano–robot.  
Se implementó un servicio llamado `/rag/ask` dentro de un nodo de ROS 2 que recibe preguntas del usuario y devuelve respuestas generadas por el modelo.

### Estructura del nodo
- **Nodo:** `rag_service`
- **Servicio:** `/rag/ask`
- **Interfaz:** `Ask.srv`
- **Lenguaje:** Python (`rclpy`)

### Flujo
1. El usuario envía una pregunta al servicio `/rag/ask` desde un cliente ROS 2.  
2. El nodo ejecuta el script `rag_pg.py`, que procesa la consulta mediante el pipeline RAG (query rewriting, hybrid retrieval, text caching)
3. El resultado se devuelve al cliente como una respuesta en texto.

### Ejemplo de Interacción
```bash
ros2 run hri_rag rag_service
```
cliente
```
ros2 service call /rag/ask hri_rag/srv/Ask "{question: '¿Quien es Fabricio?'}"
```
respuesta
```
Fabricio Banda Hernández, tiene 20 años, nació el 13 de julio del 2005 y es de Ciudad Victoria. Actualmente estudia Ingeniería en Robótica y Sistemas Digitales en el Tecnológico de Monterrey, cursando el quinto semestre.
```


Proyecto desarrollado por **Fabricio Banda Hernández**  
Reto avanzados **HRI — Roborregos 2025**
