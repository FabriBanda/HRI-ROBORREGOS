# HRI RAG — Candidates Avanzados 2025

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** en Python  
para el reto de **Human-Robot Interaction (HRI)** de Roborregos Candidates 2025.

---

## Requisitos

- [Python](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [API key de OpenAI](https://platform.openai.com/)

## Librerias principales usadas

- **openai** - para embeddings y generación de respuestas con LLM.  
- **psycopg2-binary** - para conexión con Postgres + pgvector.  
- **python-dotenv** - para cargar variables de entorno desde `.env`.  
- **rank-bm25** - implementación de BM25 para recuperación léxica.  
- **re** (estándar) - para tokenización básica.

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
- **Embeddings (pgvector)**: entienden el significado de la consulta (paráfrasis, sinónimos).
- **BM25 (léxico)**: se enfoca en coincidencias exactas (años, fechas, nombres).

### ¿Por qué BM25 y no solo TF-IDF?
TF-IDF solo cuenta ocurrencias de un término, sin fijarse en la longitud del documento ni en la saturación de palabras.  

**BM25 soluciona esto penalizando documentos largos y aplicando rendimientos decrecientes a repeticiones**, lo que asegura que el sistema recupere fragmentos más concisos y relevantes.  

Por eso use **Hybrid Retrieval junto a embeddings**, fusionando ambos rankings con **Reciprocal Rank Fusion (RRF)** para obtener la mejor respuesta combinando ambos add-ons para el reto.
## Autor

## 3. Text Caching 
Este sistema implementa **caching** en dos niveles para reducir latencia y costo de tokens.

### Caché de embeddings (`lru_cache`)
Cada vez que se genera un embedding para una consulta (`q_eff`), se guarda en memoria con `lru_cache`.  
Si la misma pregunta se repite, el vector se devuelve al instante sin llamar a OpenAI:

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

Proyecto desarrollado por **Fabricio Banda Hernández**  
Reto avanzados **HRI — Roborregos 2025**
