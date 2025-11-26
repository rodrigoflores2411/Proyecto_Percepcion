import json, base64
import cv2
import numpy as np
import pandas as pd

from src.config import WAREHOUSE_DIR, MODELS_DIR, LBPH_MODEL_PATH, LABELS_PATH

def b64png_to_gray(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(WAREHOUSE_DIR / "faces.parquet")
    if df.empty:
        raise RuntimeError("faces.parquet está vacío. Corre spark_ingest.py primero.")

    personas = sorted(df["label"].unique().tolist())
    id_map = {i: p for i, p in enumerate(personas)}
    label_of = {p: i for i, p in id_map.items()}

    images, labels = [], []
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    for _, row in df.iterrows():
        img = b64png_to_gray(row["image"])
        if img is None:
            continue
        images.append(img)
        labels.append(label_of[row["label"]])
    if not images:
        raise RuntimeError("No se reconstruyeron imágenes. Revisa faces.parquet.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.save(str(LBPH_MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print(f"[OK] Entrenado {len(personas)} personas con {len(images)} muestras.")
    print(f"[OK] Modelo: {LBPH_MODEL_PATH}")
    print(f"[OK] Labels: {LABELS_PATH}")

if __name__ == "__main__":
    main()
