
import os, sys, psycopg2
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from functools import lru_cache
import re
import datetime as _dt
import re as _re
import unicodedata

INTENT_SKILL = "skill"
INTENT_KB    = "kb"
INTENT_LLM   = "llm"

# Palabras clave skills
TIME_KEYS  = {"hora", "time"}
DATE_KEYS  = {"fecha", "que día", "que dia", "date","hoy"}
CALC_REGEX = _re.compile(r"^\s*([0-9+\-*/().\s]+)\s*$")  
MONTH_KEYS  = {"mes", "mes actual", "en que mes", "¿en qué mes", "que mes", "qué mes"}
YEAR_KEYS   = {"año", "año actual", "en qué año", "en que año", "year"}
CALC_KEYS = {"calc", "calcular", "calcula"}

BM25 = None
BM25_DOCS = []   
BM25_TOKENS = []    
BM25_META = []   
TOKEN_SPLIT = re.compile(r"\w+", flags=re.UNICODE)


SCORE_KEYS  = {"marcador", "resultado", "score", "anotación", "anotacion"}
OPPONENT_HINTS = {"chihuahua", "guadalajara", "ciudad de méxico", "ciudad de mexico", "cdmx", "monterrey"}
OPPONENT_ALIASES = {
    "chihuahua": ["chihuahua"],
    "guadalajara": ["guadalajara"],
    "ciudad de mexico": ["ciudad de mexico", "ciudad de méxico", "cdmx", "mexico city"],
    "monterrey": ["monterrey","mty"],
}


def _norm_text(s: str) -> str:

    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _guess_opponent(q: str):
    qn = _norm_text(q)
    for canon, aliases in OPPONENT_ALIASES.items():
        if any(alias in qn for alias in aliases):
            return canon  # regresamos la forma canónica
    return None



def _contains_any(text: str, keys: set[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keys)


def detect_intent(user_q: str) -> str:
    q = user_q.strip().lower()

    # skills 
    if _contains_any(q, TIME_KEYS):   return INTENT_SKILL
    if _contains_any(q, DATE_KEYS):   return INTENT_SKILL
    if _contains_any(q, MONTH_KEYS):  return INTENT_SKILL
    if _contains_any(q, YEAR_KEYS):   return INTENT_SKILL
    if _contains_any(q, SCORE_KEYS) or _guess_opponent(q) is not None:
        return INTENT_SKILL
    if _contains_any(q, CALC_KEYS) or CALC_REGEX.match(q):
        return INTENT_SKILL

    # conversación simple con LLM 
    if any(p in q for p in ["hola", "gracias", "cómo estás", "como estas", "quién eres", "quien eres"]):
        return INTENT_LLM

    # 3) rag
    return INTENT_KB

#hora
def skill_time():
    now = _dt.datetime.now()
    return f"Son las {now.strftime('%H:%M:%S')}"


#dia
def skill_date():
    today = _dt.date.today()
    # Ej: lunes 13/01/2025
    return f"Hoy es {today.strftime('%A %d/%m/%Y')}"

#año
def skill_year():
    today = _dt.date.today()
    return f"Estamos en el año {today.year}"

#mes
def skill_month():

    today = _dt.date.today()
    return f"Estamos en el mes de {today.strftime('%B').capitalize()}"
    
# calculadora 
def skill_calc(expr: str) -> str:

    if not CALC_REGEX.match(expr):
        return "Expresión no válida."
    try:
        val = eval(expr, {"__builtins__": None}, {})
        return f"= {val}"
    except Exception:
        return "No se pudo calcular esa expresión."


# patron para “xx–yy”
SCORE_PAT = _re.compile(r"(\d{1,3})\s*[–—-]\s*(\d{1,3})")

def _extract_score(text: str):
    m = SCORE_PAT.search(text)
    if not m:
        return None
    a, b = m.groups()
    return f"{a}–{b}"

def skill_score(user_q: str) -> str:
    """
    Recupera el marcador desde Postgres con regex.
    - Normaliza rival (acentos/alias)
    - Permite filtrar por etapa (final/semifinal)
    - Parametriza SIEMPRE el SQL (nada de % literales)
    """
    opp_canon = _guess_opponent(user_q)

    # base: documentos que mencionen marcador/resultado
    where_clauses = ["(chunk ILIKE %s OR chunk ILIKE %s)"]
    params = ["%marcador%", "%resultado%"]

    # si detectamos rival: agregamos todos sus alias como OR
    if opp_canon:
        aliases = OPPONENT_ALIASES[opp_canon]
        sub = "(" + " OR ".join(["chunk ILIKE %s"] * len(aliases)) + ")"
        where_clauses.append(sub)
        params.extend([f"%{a}%" for a in aliases])

    where = " AND ".join(where_clauses)

    sql = f"""
        SELECT chunk, source, chunk_idx
        FROM documents
        WHERE {where}
        ORDER BY id
        LIMIT 20
    """

    rows = []
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    for chunk, source, idx in rows:
        score = _extract_score(chunk)
        if score:
            pretty_opp = f" contra {opp_canon.title()}" if opp_canon else ""
            return f"Marcador{pretty_opp}: {score}."

    #  si se dice solo el equipo (“marcador/resultado”)
    if not rows and (opp_canon or stage):
        where2 = []
        params2 = []
        if opp_canon:
            aliases = OPPONENT_ALIASES[opp_canon]
            where2.append("(" + " OR ".join(["chunk ILIKE %s"] * len(aliases)) + ")")
            params2.extend([f"%{a}%" for a in aliases])

        if where2:
            sql2 = f"""
                SELECT chunk, source, chunk_idx
                FROM documents
                WHERE {" AND ".join(where2)}
                ORDER BY id
                LIMIT 20
            """
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql2, tuple(params2))
                    rows = cur.fetchall()
            for chunk, source, idx in rows:
                score = _extract_score(chunk)
                if score:
                    pretty_opp = f" contra {opp_canon.title()}" if opp_canon else ""
                    return f"Marcador{pretty_opp}: {score}."

    return "No encontré un marcador en los documentos."

def run_skill(user_q: str) -> str:
    q = user_q.lower()
    if _contains_any(q, TIME_KEYS):  return skill_time()
    if _contains_any(q, DATE_KEYS):  return skill_date()
    if _contains_any(q, MONTH_KEYS): return skill_month()
    if _contains_any(q, YEAR_KEYS):  return skill_year()
    if _contains_any(q, SCORE_KEYS) or _guess_opponent(q) is not None:
        return skill_score(user_q)
    if CALC_REGEX.match(q):          return skill_calc(q)
    return "Skill no encontrada."    




@lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple:
    return tuple(client.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding)

# cache respuestas finales
ANSWERS_CACHE: dict[str, str] = {}

def _answer_cache_key(q_eff: str, k: int) -> str:
    return f"k={k}|{q_eff.strip().lower()}"


def _tokenize(text: str):
    return [t.lower() for t in TOKEN_SPLIT.findall(text)]

def build_bm25_from_db():
    """Carga todos los chunks de Postgres y construye el índice BM25"""
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
    # top-k indices por score 
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
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true" 


# si no hay API key, salir
if not API_KEY:
    print(f"[ERROR] Falta OPENAI_API_KEY en {ENV_PATH}")
    sys.exit(1)


client = OpenAI(api_key=API_KEY)

# query Rewriting
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

    # ranks 1 hasta n
    sem_rank = { key_of(d): r for r, d in enumerate(sem, start=1) }
    lex_rank = { key_of(d): r for r, d in enumerate(lex, start=1) }

    keys = set(sem_rank) | set(lex_rank)
    kRR = 60.0
    fused = []
    for ky in keys:
        rrf = 0.0
        if ky in sem_rank: rrf += 1.0 / (kRR + sem_rank[ky])
        if ky in lex_rank: rrf += 1.0 / (kRR + lex_rank[ky])
       
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
    if ENABLE_CACHE:
        return list(_embed_cached(text))
    # sin cache
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
    q_eff = rewrite_query(query) or query
    print(f"[Rewriting] '{query}' ->  '{q_eff}'")

    # HIT de cache de respuesta
    if ENABLE_CACHE:
        key = _answer_cache_key(q_eff, max(k, 4))
        if key in ANSWERS_CACHE:
            return ANSWERS_CACHE[key] + "\n\n[cache-hit]"

    # recuperacion normal
    ctx = hybrid_retrieve(q_eff, k=max(k, 4))
    if not ctx:
        return "No encontre contexto en la DB"

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
    out = resp.choices[0].message.content + f"\n\n[Fuentes: {cites}]"

    # SET de cache de respuesta
    if ENABLE_CACHE:
        ANSWERS_CACHE[key] = out

    return out

def answer_with_routing(user_q: str, k: int = 4) -> str:
    intent = detect_intent(user_q)

    if intent == INTENT_SKILL:
        # las skills no usan rewriting ni RAG
        return run_skill(user_q)

    # si se detecta LLM, se responde directamente con el modelo
    if intent == INTENT_LLM:
        prompt = f"Responde brevemente y de forma amable:\n\nUsuario: {user_q}\nRespuesta:"
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return resp.choices[0].message.content

    # por defecto se obtendra del KB (RAG híbrido con rewriting + BM25 + pgvector + caching)
    return answer(user_q, k=k)

def clear_caches():
    _embed_cached.cache_clear()
    ANSWERS_CACHE.clear()
    print("[cache] Embeddings y respuestas limpiadas.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--oneshot":
        q = sys.stdin.read().strip()
        print(answer_with_routing(q, k=4))
        sys.exit(0)

    build_bm25_from_db()
    print("RAG (Hybrid + Task Distinction) listo. Escribe 'quit' para salir. Comandos: :clear\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() == "quit":
            break
        if q.strip() == ":clear":
            clear_caches()
            continue
        print("→", answer_with_routing(q, k=4), "\n")

        