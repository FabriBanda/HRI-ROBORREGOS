# HRI RAG — Candidates Avanzados 2025

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** en Python  
para el reto de **Human-Robot Interaction (HRI)** de Roborregos Candidates 2025.

---

## Requisitos

- [Python](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [API key de OpenAI](https://platform.openai.com/)

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

## Autor

Proyecto desarrollado por **Fabricio Banda Hernández**  
Reto avanzados **HRI — Roborregos 2025**
