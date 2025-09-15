# simple_rag.py — RAG con texto plano desde /kb/*.txt
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np, os, glob, sys

# Cargar API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Falta OPENAI_API_KEY en .env")
    sys.exit(1)
client = OpenAI(api_key=api_key)

# Leer todos los archivos de la carpeta kb/
files = sorted(glob.glob("kb/*.txt"))
if not files:
    print("No se encontraron archivos en kb/*.txt. Crea al menos uno (ej: kb/kb_personal.txt).")
    sys.exit(1)

kb = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Partir por doble salto de línea → “chunks” simples
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    kb.extend(chunks)

print(f"KB cargada con {len(kb)} fragmentos de {len(files)} archivos.")

# Pre-embed de la KB
def embed(text, model="text-embedding-3-small"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

kb_embeddings = [embed(doc) for doc in kb]

def retrieve(query, top_k=2):
    q_emb = np.array(embed(query)).reshape(1, -1)
    kb_vecs = np.vstack(kb_embeddings)
    sims = cosine_similarity(q_emb, kb_vecs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [kb[i] for i in top_idx], [float(sims[i]) for i in top_idx]

def rag_answer(query, top_k=2):
    ctx, scores = retrieve(query, top_k=top_k)
    context = "\n\n".join(ctx)
    prompt = f"""Usa SOLO el siguiente contexto para responder. Si falta info, dilo.
Contexto:
{context}

Pregunta: {query}
Respuesta:"""
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print("RAG con carpeta /kb listo. Escribe 'quit' para salir.\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() == "quit":
            break
        print("→", rag_answer(q, top_k=2), "\n")