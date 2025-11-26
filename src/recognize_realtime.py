import csv
import json
import os
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from io import BytesIO

from src.config import (MODELS_DIR, LBPH_MODEL_PATH, LABELS_PATH, LOGS_DIR,
                       ACCESS_LOG_CSV, ACCESS_LOG_PARQUET, TZ,
                       FACE_SIZE, DETECT_SCALE, DETECT_NEIGH, DETECT_MINSZ,
                       CONF_THRESH)

def ensure_dirs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")

def read_prev_hash():
    if not ACCESS_LOG_CSV.exists():
        return "GENESIS"
    try:
        with open(ACCESS_LOG_CSV, "rb") as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last = f.readline().decode("utf-8").strip()
        parts = last.split(",")
        return parts[-1] if len(parts) >= 1 else "GENESIS"
    except Exception:
        return "GENESIS"

def row_hash(prev_hash, row_dict):
    # Concatena campos clave para hash estable
    ordered = ["timestamp","persona","resultado","confianza","latencia_ms","fps","audio_path"]
    s = prev_hash + "|" + "|".join(str(row_dict.get(k,"")) for k in ordered)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def log_access_to_excel(name, result, confidence, image_path):
    # Solo registrar si el acceso fue concedido
    if result != "ACCESO CONCEDIDO":
        return  # No hacer nada si el acceso fue denegado

    # Ruta del archivo Excel
    log_file = 'logs/access_log.xlsx'

    # Si el archivo no existe, crear uno nuevo
    if not os.path.exists(log_file):
        wb = Workbook()
        ws = wb.active
        ws.title = "Accesos"
        # Encabezados
        ws.append(["Nombre", "Resultado", "Confianza", "Hora", "Imagen"])
        wb.save(log_file)

    # Hora actual
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Abrir el archivo Excel y agregar los registros
    wb = Workbook()
    ws = wb.active

    # Crear una imagen con la foto de la persona
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.png', image)
    img_byte_arr = BytesIO(buffer)

    # Convertir la imagen en base64 y agregarla como imagen en Excel
    img = Image(img_byte_arr)
    ws.add_image(img, 'E2')

    # Agregar datos al archivo Excel
    ws.append([name, result, confidence, current_time, img_byte_arr])

    wb.save(log_file)


def append_logs(row):
    # CSV con hash-chain
    exists = ACCESS_LOG_CSV.exists()
    with open(ACCESS_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","persona","resultado","confianza","latencia_ms","fps","audio_path","record_hash"])
        w.writerow([row[k] for k in ["timestamp","persona","resultado","confianza","latencia_ms","fps","audio_path","record_hash"]])
    # Parquet (acumulado simple)
    df = pd.DataFrame([row])
    if ACCESS_LOG_PARQUET.exists():
        old = pd.read_parquet(ACCESS_LOG_PARQUET)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(ACCESS_LOG_PARQUET, index=False)

def main():
    ensure_dirs()

    # Cargar modelo y labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(LBPH_MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        id_map = {int(k): v for k,v in json.load(f).items()}

    # Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (índice 0).")

    print("Reconocimiento en vivo — presiona 'q' para salir.")
    prev_hash = read_prev_hash()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = cv2.getTickCount()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, DETECT_SCALE, DETECT_NEIGH, minSize=DETECT_MINSZ)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, FACE_SIZE)

            label_id, confidence = recognizer.predict(roi)
            name = id_map.get(label_id, "Desconocido")
            if confidence <= CONF_THRESH:
                text = f"ACCESO CONCEDIDO: {name}"
                color = (0, 200, 0)
                result = "ACCESO_CONCEDIDO"
            else:
                text = "ACCESO DENEGADO"
                color = (0, 0, 255)
                name = "Desconocido"
                result = "ACCESO_DENEGADO"

            # Métricas de performance
            dt_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
            fps = 1000.0 / max(dt_ms, 1e-6)
            # Log con hash-chain
            row = {
                "timestamp": now_ts(),
                "persona": name,
                "resultado": result,
                "confianza": round(float(confidence), 2),
                "latencia_ms": round(float(dt_ms), 2),
                "fps": round(float(fps), 2),
                "audio_path": "",  # No se usa en este contexto
            }
            rec_hash = row_hash(prev_hash, row)
            row["record_hash"] = rec_hash
            append_logs(row)
            prev_hash = rec_hash

            # Log solo para accesos concedidos
            if result == "ACCESO_CONCEDIDO":
                log_access_to_excel(name, result, confidence, image_path=f"image_{name}.png")  # Asegúrate de tener la ruta correcta de la imagen

            # Dibujar
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"conf:{row['confianza']}  fps:{row['fps']}", (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Reconocimiento Facial (LBPH)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Finalizado.")

if __name__ == "__main__":
    main()
