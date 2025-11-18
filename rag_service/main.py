# main.py — RAG FastAPI (LangChain 1.0.0) + HF Embeddings OFFLINE + Chroma + LLM Ollama
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# LangChain (1.0.0+) — imports modernes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import Chroma -- depreciated
from langchain_chroma import Chroma

# LLM Ollama (local) - daemon
from langchain_ollama import ChatOllama

# ─────────────────────────────────────────────────────────────
# Config OFFLINE HuggingFace
# ─────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─────────────────────────────────────────────────────────────
# Paramètres (surchargeables par variables d'env)
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_DIR = Path(os.getenv("DB_DIR", "chroma_db"))
DB_COLLECTION = os.getenv("DB_COLLECTION", "kb_default")

# Chemin local du modèle HF (copié depuis une autre machine)
# ex: "C:\\hf_models\\intfloat-multilingual-e5-base"
LOCAL_HF_EMB_PATH = Path(os.getenv("LOCAL_HF_EMB_PATH", "./hf_models/intfloat-multilingual-e5-base"))
EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")  # "cpu" pas le choix sur thinkpad ryzen

# LLM Ollama (par défaut on vise Mistral)vv
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:7b-a1b-h")  # ex: "mistral" ou phi4-mini ou granite4:latest 
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite4:latest")  # ex: "mistral" ou phi4-mini ou granite4:latest 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Split
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="RAG API (HF Offline + Chroma + Ollama LLM)", version="1.1.0")

class Message(BaseModel):
    role: str   # "user" | "system"
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]
    top_k: int = 4
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings avec préfixes E5 requis."""
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")
    
# ─────────────────────────────────────────────────────────────
# Embeddings OFFLINE (dossier local)
# ─────────────────────────────────────────────────────────────
def _assert_model_folder(path: Path):
    if not path.exists() or not path.is_dir():
        raise RuntimeError(
            f"[HF OFFLINE] Dossier modèle introuvable : {path}\n"
            "→ Copiez localement le dossier complet du modèle (ex: intfloat/multilingual-e5-base)\n"
            "→ Puis définissez LOCAL_HF_EMB_PATH vers ce dossier."
        )

_assert_model_folder(LOCAL_HF_EMB_PATH)

# AVANT
# embeddings = HuggingFaceEmbeddings(
#     model_name=str(LOCAL_HF_EMB_PATH),
#     cache_folder=str(LOCAL_HF_EMB_PATH.parent),
#     model_kwargs={"device": EMB_DEVICE},
#     encode_kwargs={"batch_size": 64, "normalize_embeddings": True},  # conseillé
# )

# APRÈS
embeddings = E5Embeddings(
    model_name=str(LOCAL_HF_EMB_PATH),                 # chemin local du modèle
    cache_folder=str(LOCAL_HF_EMB_PATH.parent),
    model_kwargs={"device": EMB_DEVICE},
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True},   # conseillé avec cosine
)

# ─────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────
def get_vectorstore():
    return Chroma(
        collection_name=DB_COLLECTION,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

# ─────────────────────────────────────────────────────────────
# LLM (Ollama) — avec fallback si indisponible
# ─────────────────────────────────────────────────────────────
def build_llm():
    try:
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    except Exception as e:
        print(f"[WARN] Ollama indisponible ({e}). Le service répondra sans génération LLM.")
        return None

llm = build_llm()

SYSTEM_PROMPT = """Tu es un assistant RAG francophone.
- Appuie-toi STRICTEMENT sur le contexte fourni.
- Si l'information manque, dis-le clairement.
- Commence par 2–3 phrases de synthèse, puis liste des références [n] vers les passages utiles.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Question:\n{question}\n\n"
     "=== CONTEXTE ===\n{context}\n"
     "================\n\n"
     "Consigne: Réponds en français, clair et concis, en citant les passages avec [n].")
])

def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "inconnu")
        page = meta.get("page", None)
        tag = f"{src}" + (f" (page {page})" if page is not None else "")
        parts.append(f"[{i}] {tag}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/healthz")
def health():
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
    except Exception:
        count = None
    return {
        "status": "ok",
        "collection": DB_COLLECTION,
        "db_path": str(DB_DIR),
        "doc_count": count,
        "hf_offline": True,
        "local_hf_emb_path": str(LOCAL_HF_EMB_PATH),
        "device": EMB_DEVICE,
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
        "llm_ready": llm is not None,
    }

@app.post("/ingest")
def ingest_local_folder(
    folder: str = Form(default=str(DATA_DIR)),
    glob: str = Form(default="**/*"),
):
    folder_path = Path(folder)
    if not folder_path.exists():
        return {"ok": False, "msg": f"Dossier introuvable: {folder_path}"}

    docs: List[Document] = []
    for path in folder_path.glob(glob):
        if path.is_dir():
            continue
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in [".txt", ".md", ".log", ".csv"]:
            docs.extend(TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load())

    if not docs:
        return {"ok": False, "msg": "Aucun document chargé."}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)
   # vs.persist()

    # Optionnel: flush compatible anciennes versions
    if hasattr(vs, "_client") and hasattr(vs._client, "persist"):
        vs._client.persist()

    return {"ok": True, "added_chunks": len(chunks)}

@app.post("/ingest_file")
async def ingest_file(file: UploadFile = File(...)):
    target = DATA_DIR / file.filename
    with open(target, "wb") as f:
        f.write(await file.read())

    if target.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(target)).load()
    else:
        docs = TextLoader(str(target), encoding="utf-8", autodetect_encoding=True).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)

    # Optionnel: flush compatible anciennes versions
    if hasattr(vs, "_client") and hasattr(vs._client, "persist"):
        vs._client.persist()

    return {"ok": True, "file": file.filename, "added_chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question_parts = [m.content for m in req.messages if m.role == "user"]
    question = "\n\n".join(question_parts).strip() or "Bonjour !"

    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=req.top_k)
    context = format_docs(docs)

    if not docs:
        answer = "Je n'ai trouvé aucun passage pertinent dans la base. Ingest davantage de documents."
    else:
        if llm is None:
            # Fallback: pas de génération, renvoyer les passages
            answer = (
                "Synthèse (fallback sans LLM) : voici les passages pertinents trouvés.\n\n" + context
            )
        else:
            chain = RAG_PROMPT | llm
            result = chain.invoke({"question": question, "context": context})
            answer = result.content

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "source": meta.get("source", "inconnu"),
            "page": meta.get("page"),
            "chunk": meta.get("start_index"),
        })

    return QueryResponse(answer=answer, sources=sources)
