# app_demo_rag.py
# Front de démo Streamlit pour le microservice RAG (HF offline + Chroma + Ollama)
# Endpoints couverts : /healthz, /ingest, /ingest_file, /query

import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="RAG Demo", page_icon="🧠", layout="wide")

# -------------------------------
# Configuration de base
# -------------------------------
st.sidebar.title("⚙️ Configuration")
base_url = st.sidebar.text_input("Base URL de l'API", value="http://localhost:8000")
timeout_s = st.sidebar.number_input("Timeout (secondes)", min_value=1, max_value=600, value=30, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Ce front appelle directement les endpoints FastAPI de ton microservice RAG LangChain, Embeddings E5 Base OFFLINE + Chroma + LLM Ollama.")

# -------------------------------
# Onglets
# -------------------------------
tab_health, tab_ingest, tab_ingest_file, tab_query = st.tabs(
    ["🔍 Healthz", "📂 Ingest (dossier)", "📄 Ingest (fichier)", "💬 Query"]
)

# -------------------------------
# 1) HEALTHZ
# -------------------------------
with tab_health:
    st.header("🔍 Healthz")
    st.write("Appelle `GET /healthz` et affiche l’état de la base, du modèle d’embeddings (offline), et du LLM Ollama.")

    if st.button("Tester /healthz", type="primary"):
        try:
            resp = requests.get(f"{base_url}/healthz", timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            st.success("✅ /healthz OK")
            st.json(data)
        except Exception as e:
            st.error(f"❌ Erreur /healthz : {e}")

# -------------------------------
# 2) INGEST (dossier)
# -------------------------------
with tab_ingest:
    st.header("📂 Ingest d’un dossier")
    st.write("Appelle `POST /ingest` avec `folder` et `glob` (ex: **/*.pdf).")

    col1, col2 = st.columns([2, 1])
    with col1:
        folder = st.text_input("Chemin du dossier", value=str(Path("data").absolute()))
    with col2:
        glob = st.text_input("Glob", value="**/*")

    if st.button("Lancer l’ingest (dossier)", type="primary"):
        try:
            resp = requests.post(
                f"{base_url}/ingest",
                data={"folder": folder, "glob": glob},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            st.success("✅ Ingestion dossier terminée")
            st.json(resp.json())
        except Exception as e:
            st.error(f"❌ Erreur /ingest : {e}")

# -------------------------------
# 3) INGEST (fichier)
# -------------------------------
with tab_ingest_file:
    st.header("📄 Ingest d’un fichier unique")
    st.write("Appelle `POST /ingest_file` avec un upload (PDF, TXT, MD, CSV…).")

    uploaded = st.file_uploader("Sélectionne un fichier à ingérer", type=None)
    if st.button("Lancer l’ingest (fichier)", type="primary", disabled=(uploaded is None)):
        if uploaded is None:
            st.warning("Choisis d’abord un fichier.")
        else:
            try:
                files = {"file": (uploaded.name, uploaded.getvalue())}
                resp = requests.post(f"{base_url}/ingest_file", files=files, timeout=timeout_s)
                resp.raise_for_status()
                st.success("✅ Ingestion fichier terminée")
                st.json(resp.json())
            except Exception as e:
                st.error(f"❌ Erreur /ingest_file : {e}")

# -------------------------------
# 4) QUERY
# -------------------------------
with tab_query:
    st.header("💬 Query (RAG)")
    st.write("Appelle `POST /query` avec un array de messages. Affiche la réponse et les sources.")

    system_prompt = st.text_area(
        "Message système (optionnel)",
        value="Tu es concis et tu réponds en 10 lignes au maximum.",
        height=80
    )
    user_question = st.text_area(
        "Question",
        value="Qu'est ce qu'une data fabric ?",
        height=120
    )
    top_k = st.number_input("top_k (nombre de chunks retournés)", min_value=1, max_value=20, value=4, step=1)

    if st.button("Interroger /query", type="primary"):
        payload = {
            "messages": [],
            "top_k": int(top_k)
        }
        if system_prompt.strip():
            payload["messages"].append({"role": "system", "content": system_prompt.strip()})
        payload["messages"].append({"role": "user", "content": user_question.strip()})

        try:
            resp = requests.post(
                f"{base_url}/query",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
            st.success("✅ Réponse reçue")

            # Affichage réponse
            st.subheader("🧠 Réponse")
            st.write(data.get("answer", ""))

            # Affichage sources
            st.subheader("📚 Sources")
            sources = data.get("sources", [])
            if sources:
                df = pd.DataFrame(sources)
                # Tenter d'ordonner colonnes
                for col in ["source", "page", "chunk"]:
                    if col not in df.columns:
                        df[col] = None
                df = df[["source", "page", "chunk"]]
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune source renvoyée.")

            # Brute JSON (debug)
            with st.expander("Voir le JSON brut"):
                st.json(data)

        except Exception as e:
            st.error(f"❌ Erreur /query : {e}")
