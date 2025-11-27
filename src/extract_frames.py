import cv2
import os

def extraer_frames(video_path, persona, intervalo_segundos=0.5):

    # Verificar si la carpeta persona existe
    carpeta_persona = os.path.join("dataset", persona)
    if not os.path.exists(carpeta_persona):
        raise FileNotFoundError(f"La carpeta 'dataset/{persona}' no existe.")

    # Crear carpeta de frames
    carpeta_frames = os.path.join(carpeta_persona, "frames")
    os.makedirs(carpeta_frames, exist_ok=True)

    # Cargar video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir el video. Verifica la ruta.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    intervalo_frames = int(fps * intervalo_segundos)

    frame_count = 0
    contador_img = 1

    print(f"📹 Procesando video: {video_path}")
    print(f"🧑 Persona: {persona}")
    print(f"💾 Guardando en: {carpeta_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar imagen cada intervalo
        if frame_count % intervalo_frames == 0:
            nombre_img = os.path.join(carpeta_frames, f"{persona}_frame_{contador_img}.jpg")
            cv2.imwrite(nombre_img, frame)
            print(f"🖼 Guardado: {nombre_img}")
            contador_img += 1

        frame_count += 1

    cap.release()
    print("✅ Proceso completado. Frames extraídos correctamente.")


if __name__ == "__main__":
    # EJEMPLO DE USO:
    # Aquí colocas la ruta del video
    video = ""

    # Aquí indicas a qué subcarpeta debe ir
    persona = ""

    extraer_frames(video, persona, intervalo_segundos=0.5)
