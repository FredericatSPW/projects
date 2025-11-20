# streamlit_app.py
import os
from io import BytesIO
from typing import Dict, Any

import requests
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Image Anonymizer Demo", layout="wide")

# ---------- Config ----------
DEFAULT_API_URL = os.getenv("ANON_API_URL", "http://localhost:8001")

def ping_health(api_url: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}

def post_anonymize_image(api_url: str, file_bytes: bytes, params: Dict[str, Any]) -> bytes:
    url = f"{api_url.rstrip('/')}/anonymize"
    files = {"file": ("upload", file_bytes, "application/octet-stream")}
    data = {
        "expand_faces": str(params["expand_faces"]),
        "expand_plates": str(params["expand_plates"]),
        "draw": "true" if params["draw"] else "false",
        "face_color": params["face_color"],
        "plate_color": params["plate_color"],
        "blur_method": params["blur_method"],
        "blur_factor": str(params["blur_factor"]),
        "out_format": params["out_format"],
        "return_json": "false",
    }
    r = requests.post(url, files=files, data=data, timeout=60)
    if "application/json" in r.headers.get("content-type", ""):
        # Le service n'aurait pas dû renvoyer du JSON ici
        raise RuntimeError(f"Réponse JSON inattendue: {r.text}")
    r.raise_for_status()
    return r.content

def post_anonymize_json(api_url: str, file_bytes: bytes, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/anonymize"
    files = {"file": ("upload", file_bytes, "application/octet-stream")}
    data = {
        "expand_faces": str(params["expand_faces"]),
        "expand_plates": str(params["expand_plates"]),
        "draw": "true" if params["draw"] else "false",
        "face_color": params["face_color"],
        "plate_color": params["plate_color"],
        "blur_method": params["blur_method"],
        "blur_factor": str(params["blur_factor"]),
        "out_format": params["out_format"],
        "return_json": "true",
    }
    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------- Sidebar ----------
st.sidebar.title("API")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
health = ping_health(api_url)
if health.get("status") == "ok":
    st.sidebar.success(f"Service OK (device={health.get('device')})")
else:
    st.sidebar.error("Service indisponible")
    if "error" in health:
        st.sidebar.caption(health["error"])

st.sidebar.title("Paramètres")
expand_faces = st.sidebar.slider("Agrandissement visage", 0.0, 1.0, 0.40, 0.01)
expand_plates = st.sidebar.slider("Agrandissement plaque", 0.0, 0.5, 0.10, 0.01)
blur_method = st.sidebar.selectbox("Méthode de flou", ["box", "gaussian"])
blur_factor = st.sidebar.slider("Intensité du flou (plus petit = plus fort)", 2, 40, 4, 1)
draw = st.sidebar.checkbox("Dessiner les boîtes", True)
face_color = st.sidebar.selectbox("Couleur boîtes visage", ["jaune", "rouge", "vert", "bleu", "cyan", "magenta", "blanc", "noir"], index=0)
plate_color = st.sidebar.selectbox("Couleur boîtes plaque", ["rouge", "jaune", "vert", "bleu", "cyan", "magenta", "blanc", "noir"], index=0)
out_format = st.sidebar.selectbox("Format de sortie", ["jpg", "png", "webp"], index=0)
show_details = st.sidebar.checkbox("Afficher les détails de détection (JSON)")

params = {
    "expand_faces": expand_faces,
    "expand_plates": expand_plates,
    "blur_method": blur_method,
    "blur_factor": blur_factor,
    "draw": draw,
    "face_color": face_color,
    "plate_color": plate_color,
    "out_format": out_format,
}

st.sidebar.title("Caméra")
show_camera = st.sidebar.checkbox("Afficher la prise de vue caméra", value=True)

# ---------- Main UI ----------
st.title("🕶️ Image Anonymizer — Demo")
st.caption("Détection & floutage de visages et plaques via votre micro-service FastAPI.")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Entrée")
    uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg", "webp"])
    
    snap = None
    if show_camera:
        st.caption("ou")
        snap = st.camera_input("Prendre une photo (optionnel)")
        
    input_bytes = None
    input_name = None
    if uploaded is not None:
        input_bytes = uploaded.read()
        input_name = uploaded.name
    elif snap is not None:
        input_bytes = snap.getvalue()
        input_name = "camera.jpg"

    if input_bytes:
        try:
            st.image(Image.open(BytesIO(input_bytes)), caption=f"Image source — {input_name}", use_container_width=True)
        except Exception:
            st.warning("Prévisualisation impossible, mais le fichier sera envoyé au service.")

    go = st.button("Anonymiser", type="primary", use_container_width=True)

with col2:
    st.subheader("Sortie")
    if go:
        if not input_bytes:
            st.error("Aucune image fournie.")
        else:
            try:
                # Optionnel : détails JSON (boîtes, compteurs)
                details = None
                if show_details:
                    details = post_anonymize_json(api_url, input_bytes, params)

                # Image anonymisée
                out_bytes = post_anonymize_image(api_url, input_bytes, params)
                try:
                    out_img = Image.open(BytesIO(out_bytes))
                    st.image(out_img, caption="Image anonymisée", use_container_width=True)
                except Exception:
                    st.info("Réception d'un binaire non image (téléchargeable ci-dessous).")

                st.download_button(
                    "Télécharger l'image",
                    data=out_bytes,
                    file_name=f"anonymized.{out_format}",
                    mime={"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[out_format],
                    use_container_width=True,
                )

                if details:
                    st.markdown("**Détails détection (JSON)**")
                    st.json(details)

            except requests.HTTPError as e:
                # Essayer d'extraire le message d'erreur JSON du service
                msg = None
                try:
                    msg = e.response.json().get("detail")
                except Exception:
                    pass
                st.error(f"HTTP {e.response.status_code}: {msg or str(e)}")
            except Exception as e:
                st.error(f"Erreur: {e}")

st.markdown("---")
st.caption("Astuce: changez la méthode de flou, l'intensité et l'agrandissement pour montrer l'impact sur la confidentialité.")
