# ingest_pg.py
import os, glob
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/hri_rag")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  chunk TEXT NOT NULL,
  source TEXT,
  chunk_idx INT,
  metadata JSONB,
  embedding VECTOR({EMBED_DIM})
);
CREATE INDEX IF NOT EXISTS idx_documents_embedding
  ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

def embed(text: str) -> list[float]:
    return client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding

def iter_chunks():
    files = sorted(glob.glob("kb/*.txt"))
    if not files:
        raise SystemExit("No hay archivos en kb/*.txt")
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for i, p in enumerate(parts):
            yield os.path.basename(fp), i, p

def main():
    conn = psycopg2.connect(DB_URL); conn.autocommit = True
    cur = conn.cursor(); cur.execute(SCHEMA_SQL)

    cur.execute("TRUNCATE documents RESTART IDENTITY") 
    rows = 0
    for src, idx, chunk in iter_chunks():
        vec = embed(chunk)
        cur.execute(
            """
            INSERT INTO documents(chunk, source, chunk_idx, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (chunk, src, idx, None, f"[{', '.join(map(str, vec))}]")
        )
        rows += 1
    cur.close(); conn.close()
    print(f"Ingesta completa: {rows} chunks.")

if __name__ == "__main__":
    main()
    
    