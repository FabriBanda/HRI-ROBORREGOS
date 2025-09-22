# rag_pg.py (versión con .env explícito + depuración)
import os, sys, psycopg2
from contextlib import closing
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1) Cargar .env desde la misma carpeta del script (no depende del cwd)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL  = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/hri_rag")
MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"

if not API_KEY:
    print(f"[ERROR] Falta OPENAI_API_KEY en {ENV_PATH}")
    sys.exit(1)


client = OpenAI(api_key=API_KEY)

# Query Rewriting
def rewrite_query(raw_q: str, model: str = MODEL) -> str:
    """Reescribe la pregunta para búsqueda semántica (breve y sin ruido)."""
    prompt = (
    "Reescribe la siguiente pregunta para búsqueda semántica.\n"
    "- Mantén el mismo idioma.\n"
    "- Quita muletillas y ambigüedad.\n"
    "- Si la pregunta ya es clara y concisa, devuélvela igual sin cambios.\n"
    f"\nPregunta: {raw_q}\nReescrita:"
)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un reescritor de consultas para RAG."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        out = resp.choices[0].message.content.strip()
        # fallback si el modelo devuelve algo vacío o muy corto sin sentido
        return out if len(out) > 0 else raw_q
    except Exception:
        return raw_q  


def embed(text: str) -> list[float]:
    return client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding

def retrieve(query: str, k: int = 3):
    qvec = embed(query)
    qvec_sql = f"[{', '.join(map(str, qvec))}]"

    sql = f"""
        SELECT chunk, source, chunk_idx, (embedding <-> '{qvec_sql}'::vector) AS l2
        FROM documents
        ORDER BY embedding <-> '{qvec_sql}'::vector
        LIMIT {int(k)}
    """

    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return [{"chunk": r[0], "source": r[1], "idx": r[2], "sim": float(-r[3])} for r in rows]

def answer(query: str, k: int = 3) -> str:
    q_eff = rewrite_query(query)  # rewriting 
    print(f"[Rewriting] '{query}' ->  '{q_eff}'") 

    ctx = retrieve(q_eff, k)
    if not ctx:
        return "No encontré contexto en la DB. ¿Ya corriste ingest_pg.py?"

    context = "\n\n".join([c["chunk"] for c in ctx])
    prompt = f"""Responde SOLO con el CONTEXTO. Si falta informacion, dilo.
CONTEXTO:
{context}

PREGUNTA (original): {query}
PREGUNTA (reescrita): {q_eff}
RESPUESTA:"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    cites = " | ".join([f"{c['source']}#chunk{c['idx']} (sim={c['sim']:.2f})" for c in ctx])
    return resp.choices[0].message.content + f"\n\n[Fuentes: {cites}]"

if __name__ == "__main__":
    print("RAG con Postgres+pgvector. Escribe 'quit' para salir.\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() == "quit":
            break
        print(answer(q, k=3), "\n")