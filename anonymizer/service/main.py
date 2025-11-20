import os
from io import BytesIO
from typing import Optional, Literal, Dict, Any, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO


def ensure_exists(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier de modèle introuvable pour {label}: {path}. "
            f"Place-le dans ./models ou passe un chemin via variable d'environnement."
        )
    return path


def load_models() -> Dict[str, Any]:
    face_path = ensure_exists(
        os.getenv("FACE_MODEL_PATH", "models/yolov8n-face-lindevs.pt"), "visage"
    )
    plate_path = ensure_exists(
        os.getenv("PLATE_MODEL_PATH", "models/license_plate_detector.pt"), "plaque"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    face_model = YOLO(face_path)
    plate_model = YOLO(plate_path)

    return {"face": face_model, "plate": plate_model, "device": device}


def read_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier image vide.")
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Impossible de décoder l'image.")
    return img


def to_bgr_tuple(color_name: str) -> tuple:
    """
    Convertit un nom simple en BGR (OpenCV). Support minimal pour éviter
    la config verbeuse côté client.
    """
    name = color_name.lower()
    palette = {
        "rouge": (0, 0, 255),
        "jaune": (0, 255, 255),
        "vert": (0, 255, 0),
        "bleu": (255, 0, 0),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "blanc": (255, 255, 255),
        "noir": (0, 0, 0),
    }
    return palette.get(name, (0, 0, 255))


def expand_box(box: np.ndarray, image_shape: tuple, scale: float) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    dw = int(w * scale / 2)
    dh = int(h * scale / 2)
    x1n = max(0, x1 - dw)
    y1n = max(0, y1 - dh)
    x2n = min(image_shape[1] - 1, x2 + dw)
    y2n = min(image_shape[0] - 1, y2 + dh)
    return [x1n, y1n, x2n, y2n]


def blur_regions(
    image: np.ndarray,
    boxes: List[List[int]],
    method: Literal["box", "gaussian"] = "box",
    blur_factor: int = 4,
) -> np.ndarray:
    """
    blur_factor = 4 -> noyau ~ taille_boite/4 (min 5)
    """
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = max((x2 - x1), (y2 - y1)) // max(1, blur_factor)
        k = max(5, k)  # force un minimum
        if method == "gaussian":
            # noyau impair requis
            if k % 2 == 0:
                k += 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            blurred = cv2.blur(roi, (k, k))
        image[y1:y2, x1:x2] = blurred
    return image


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    color=(0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image


def detect_boxes(model: YOLO, img: np.ndarray, device: str) -> List[List[int]]:
    results = model(img, device=device, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return []
    boxes = results.boxes.xyxy
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    return boxes.astype(int).tolist()


def encode_image(img: np.ndarray, fmt: Literal["jpg", "png", "webp"]) -> bytes:
    ext_map = {"jpg": ".jpg", "png": ".png", "webp": ".webp"}
    ext = ext_map.get(fmt, ".jpg")
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise HTTPException(status_code=500, detail="Échec d'encodage de l'image.")
    return buf.tobytes()


app = FastAPI(title="Image Anonymizer API", version="1.0.0")

# CORS large par défaut (ajuste origins si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}  # chargé au startup


@app.on_event("startup")
def _startup():
    global MODELS
    MODELS = load_models()
    print(
        f"[startup] Modèles chargés (device={MODELS['device']}). "
        f"Visage: {os.getenv('FACE_MODEL_PATH', 'models/yolov8n-face-lindevs.pt')} | "
        f"Plaque: {os.getenv('PLATE_MODEL_PATH', 'models/license_plate_detector.pt')}"
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": MODELS.get("device", "cpu")}


@app.post(
    "/anonymize",
    responses={
        200: {
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "image/webp": {},
                "application/json": {},
            },
            "description": "Image anonymisée ou JSON (selon return_json)",
        }
    },
)
def anonymize(
    file: UploadFile = File(..., description="Image à anonymiser"),
    expand_faces: float = Form(0.4, description="Agrandissement des boîtes visage"),
    expand_plates: float = Form(0.10, description="Agrandissement des boîtes plaque"),
    draw: bool = Form(True, description="Dessiner les boîtes sur l'image finale"),
    face_color: str = Form("jaune", description="Couleur des boîtes visage"),
    plate_color: str = Form("rouge", description="Couleur des boîtes plaque"),
    blur_method: Literal["box", "gaussian"] = Form("box"),
    blur_factor: int = Form(4, description="Plus petit = flou plus fort"),
    out_format: Literal["jpg", "png", "webp"] = Form("jpg"),
    return_json: bool = Form(False, description="Si vrai, renvoie JSON plutôt qu'image"),
):
    """
    Envoie une image et récupère l'image anonymisée (ou les méta-données en JSON).
    """
    try:
        img = read_image_from_upload(file)
    finally:
        # Réinitialise le pointeur si quelqu'un veut relire le stream plus tard
        try:
            file.file.seek(0)
        except Exception:
            pass

    # Détection
    faces_raw = detect_boxes(MODELS["face"], img, MODELS["device"])
    plates_raw = detect_boxes(MODELS["plate"], img, MODELS["device"])

    # Expand
    faces = [expand_box(np.array(b), img.shape, expand_faces) for b in faces_raw]
    plates = [expand_box(np.array(b), img.shape, expand_plates) for b in plates_raw]

    # Flou
    out = img.copy()
    out = blur_regions(out, faces, method=blur_method, blur_factor=blur_factor)
    out = blur_regions(out, plates, method=blur_method, blur_factor=blur_factor)

    # Dessin (optionnel)
    if draw:
        out = draw_boxes(out, faces, color=to_bgr_tuple(face_color), thickness=2)
        out = draw_boxes(out, plates, color=to_bgr_tuple(plate_color), thickness=2)

    if return_json:
        return JSONResponse(
            {
                "faces_detected": len(faces_raw),
                "plates_detected": len(plates_raw),
                "faces_boxes": faces,
                "plates_boxes": plates,
                "params": {
                    "expand_faces": expand_faces,
                    "expand_plates": expand_plates,
                    "blur_method": blur_method,
                    "blur_factor": blur_factor,
                    "draw": draw,
                },
            }
        )

    # Image binaire
    data = encode_image(out, out_format)
    media = {
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }[out_format]
    return StreamingResponse(BytesIO(data), media_type=media)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
