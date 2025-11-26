import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from collections import Counter

DATASET_DIR = "dataset"        
FACE_SIZE = (200, 200)         
DETECT_SCALE = 1.1
DETECT_NEIGH = 4
DETECT_MINSZ = (30, 30)
CONF_THRESH = 100              
def load_dataset():
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for person_folder in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_folder)
        if os.path.isdir(person_path):
            if person_folder not in label_map:
                label_map[person_folder] = label_counter
                label_counter += 1
            for image_file in os.listdir(person_path):
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    img_path = os.path.join(person_path, image_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, FACE_SIZE)
                        images.append(img)
                        labels.append(label_map[person_folder])
    return np.array(images), np.array(labels), label_map

def evaluate_model(images, labels):
    if len(np.unique(labels)) < 2:
        print("No hay suficientes clases para partición estratificada, se usará train_test_split sin estratificación.")
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, stratify=labels, random_state=42
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    start_train = time.time()
    recognizer.train(list(X_train), y_train.astype(np.int32))
    end_train = time.time()
    train_time = end_train - start_train

    preds = []
    latencias = []
    fps_list = []

    for img in X_test:
        t0 = time.time()
        label_pred, confidence = recognizer.predict(img)
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        fps = 1000.0 / max(latency_ms, 1e-6)
        latencias.append(latency_ms)
        fps_list.append(fps)
        preds.append(label_pred)

    accuracy = accuracy_score(y_test, preds)
    return {
        "accuracy": accuracy,
        "latency_ms": np.mean(latencias),
        "fps": np.mean(fps_list),
        "train_time": train_time
    }
def cross_validation(images, labels, k=5):
    num_classes = len(np.unique(labels))
    if num_classes < 2:
        print("No hay suficientes clases para validación cruzada, omitiendo CV.")
        return None

    # Ajustar k según mínimo de muestras por clase
    from collections import Counter
    min_samples_per_class = min(Counter(labels).values())
    k = min(k, min_samples_per_class)
    if k < 2:
        print("Demasiadas pocas imágenes por clase para CV, omitiendo CV.")
        return None

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies, latencies, fps_list = [], [], []

    for train_idx, test_idx in skf.split(images, labels):
        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(list(X_train), y_train.astype(np.int32))  

        preds = []
        fold_latencies, fold_fps = [], []

        for img in X_test:
            t0 = time.time()
            label_pred, confidence = recognizer.predict(img)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000
            fps = 1000.0 / max(latency_ms, 1e-6)
            fold_latencies.append(latency_ms)
            fold_fps.append(fps)
            preds.append(label_pred)

        accuracies.append(accuracy_score(y_test, preds))
        latencies.append(np.mean(fold_latencies))
        fps_list.append(np.mean(fold_fps))

    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "fps_mean": np.mean(fps_list),
        "fps_std": np.std(fps_list)
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    images, labels, label_map = load_dataset()
    print(f"Dataset cargado: {len(images)} imágenes, {len(label_map)} personas.")

    print("Evaluando modelo con partición 80/20...")
    metrics = evaluate_model(images, labels)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Latencia promedio: {metrics['latency_ms']:.2f} ms")
    print(f"FPS promedio: {metrics['fps']:.2f}")
    print(f"Tiempo de entrenamiento: {metrics['train_time']:.2f} s")

    print("\nRealizando validación cruzada 5-fold...")
    cv_metrics = cross_validation(images, labels, k=5)
    if cv_metrics is not None:
        print(f"Accuracy CV: {cv_metrics['accuracy_mean']*100:.2f}% ± {cv_metrics['accuracy_std']*100:.2f}%")
        print(f"Latencia CV: {cv_metrics['latency_mean']:.2f} ± {cv_metrics['latency_std']:.2f} ms")
        print(f"FPS CV: {cv_metrics['fps_mean']:.2f} ± {cv_metrics['fps_std']:.2f}")
    else:
        print("Validación cruzada omitida por insuficiencia de datos.")

    # ----------------------------
    # Visualización
    # ----------------------------
    plt.figure(figsize=(8,5))
    plt.bar(["Train/Test Split", "Cross-Validation" if cv_metrics else "N/A"],
            [metrics['accuracy'], cv_metrics['accuracy_mean'] if cv_metrics else 0],
            yerr=[0, cv_metrics['accuracy_std'] if cv_metrics else 0], alpha=0.7, color=["green","blue"])
    plt.ylabel("Accuracy")
    plt.title("Precisión del modelo LBPH")
    plt.ylim(0,1)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(["Train/Test Split", "Cross-Validation" if cv_metrics else "N/A"],
            [metrics['latency_ms'], cv_metrics['latency_mean'] if cv_metrics else 0],
            yerr=[0, cv_metrics['latency_std'] if cv_metrics else 0], alpha=0.7, color=["orange","red"])
    plt.ylabel("Latencia (ms)")
    plt.title("Latencia del reconocimiento facial")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(["Train/Test Split", "Cross-Validation" if cv_metrics else "N/A"],
            [metrics['fps'], cv_metrics['fps_mean'] if cv_metrics else 0],
            yerr=[0, cv_metrics['fps_std'] if cv_metrics else 0], alpha=0.7, color=["purple","cyan"])
    plt.ylabel("FPS")
    plt.title("FPS del reconocimiento facial")
    plt.show()
