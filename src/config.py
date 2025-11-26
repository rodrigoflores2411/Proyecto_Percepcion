from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
WAREHOUSE_DIR = ROOT / "warehouse"

LBPH_MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABELS_PATH = MODELS_DIR / "labels.json"
ACCESS_LOG_CSV = LOGS_DIR / "access_log.csv"
ACCESS_LOG_PARQUET = LOGS_DIR / "access_log.parquet"
AUDIO_DIR = LOGS_DIR / "audio"
TZ = "America/Mexico_City" 
FACE_SIZE = (100, 100)
DETECT_SCALE = 1.2
DETECT_NEIGH = 5
DETECT_MINSZ = (60, 60)
CONF_THRESH = 60.0