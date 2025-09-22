
import os, sys, psycopg2
from contextlib import closing
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import re

BM25 = None
BM25_DOCS = []      # textos (chunks)
BM25_TOKENS = []    # tokens por doc
BM25_META = []      # (source, idx)
TOKEN_SPLIT = re.compile(r"\w+", flags=re.UNICODE)

def _tokenize(text: str):
    return [t.lower() for t in TOKEN_SPLIT.findall(text)]

def build_bm25_from_db():
    """Carga todos los chunks de Postgres y construye el índice BM25."""
    global BM25, BM25_DOCS, BM25_TOKENS, BM25_META
    BM25_DOCS.clear(); BM25_TOKENS.clear(); BM25_META.clear()
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk, source, chunk_idx FROM documents ORDER BY id")
            rows = cur.fetchall()
    for chunk, source, idx in rows:
        BM25_DOCS.append(chunk)
        BM25_TOKENS.append(_tokenize(chunk))
        BM25_META.append((source, idx))
    BM25 = BM25Okapi(BM25_TOKENS)
    print(f"[BM25] Index construido con {len(BM25_DOCS)} documentos.")
    
    
def lexical_retrieve(query: str, k: int = 3):
    if not BM25:
        return []
    q_tokens = _tokenize(query)
    scores = BM25.get_scores(q_tokens)
    # Top-k indices por score desc
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in idxs:
        chunk = BM25_DOCS[i]
        source, idx = BM25_META[i]
        out.append({"chunk": chunk, "source": source, "idx": idx, "score_bm25": float(scores[i])})
    return out

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
    "Mantén el mismo idioma.\n"
    "Quita muletillas y ambigüedad.\n"
    "Si la pregunta ya es clara y concisa, devuélvela igual sin cambios.\n"
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

def hybrid_retrieve(query: str, k: int = 4):
    
    sem_k = max(k, 4)               
    sem = retrieve(query, k=sem_k)   # lista rankeada por semantica

 
    lex = lexical_retrieve(query, k=sem_k)  # lista rankeada por BM25

    def key_of(d): return (d["source"], d["idx"])

    # ranks 1..n
    sem_rank = { key_of(d): r for r, d in enumerate(sem, start=1) }
    lex_rank = { key_of(d): r for r, d in enumerate(lex, start=1) }

    keys = set(sem_rank) | set(lex_rank)
    kRR = 60.0
    fused = []
    for ky in keys:
        rrf = 0.0
        if ky in sem_rank: rrf += 1.0 / (kRR + sem_rank[ky])
        if ky in lex_rank: rrf += 1.0 / (kRR + lex_rank[ky])
        # toma contenido desde sem o lex
        found = next((d for d in sem if key_of(d) == ky), None) or next((d for d in lex if key_of(d) == ky), None)
        fused.append({
            "chunk": found["chunk"],
            "source": found["source"],
            "idx": found["idx"],
            "sim": found.get("sim", 0.0),
            "rrf": rrf
        })
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    return fused[:k]

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
    #Reescritura de la pregunta 
    q_eff = rewrite_query(query) or query
    print(f"[Rewriting] '{query}' ->  '{q_eff}'") 
    # Se usa el hibrido del embedings y BM25
    ctx = hybrid_retrieve(q_eff, k=max(k, 4))
    if not ctx:
        return "No encontré contexto en la DB"

    context = "\n\n".join([c["chunk"] for c in ctx])
    prompt = f"""Responde SOLO con el CONTEXTO. Si la información no está clara, responde con lo más cercano posible dentro del contexto.
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
    cites = " | ".join([f"{c['source']}#chunk{c['idx']} (rrf={c['rrf']:.4f})" for c in ctx])
    return resp.choices[0].message.content + f"\n\n[Fuentes: {cites}]"


if __name__ == "__main__":
    build_bm25_from_db()
    print("RAG (Hybrid) listo. Escribe 'quit' para salir.\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() == "quit":
            break
        print("→", answer(q, k=4), "\n")