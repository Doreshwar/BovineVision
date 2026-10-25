"""
BovineVision AI - Configuration Module
"""

import os
from pydantic import BaseModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class Settings(BaseModel):
    PROJECT_NAME: str = "BovineVision AI"
    VERSION: str = "1.0.0"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
    REJECTION_MESSAGE: str = "Could not detect, try another picture"

    # Paths
    DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
    DATA_SPLITS_DIR: str = os.path.join(BASE_DIR, "data")
    SAVED_MODELS_DIR: str = os.path.join(BASE_DIR, "backend", "saved_models")
    KNOWLEDGE_BASE_DIR: str = os.path.join(BASE_DIR, "backend", "knowledge_base")
    CHECKPOINT_PATH: str = os.path.join(BASE_DIR, "backend", "saved_models", "best_model.pth")
    CLASSES_PATH: str = os.path.join(BASE_DIR, "backend", "saved_models", "classes.json")
    METRICS_PATH: str = os.path.join(BASE_DIR, "backend", "saved_models", "metrics.json")

    # Image processing
    INPUT_SIZE: int = 224
    NORMALIZE_MEAN: list = [0.485, 0.456, 0.406]
    NORMALIZE_STD: list = [0.229, 0.224, 0.225]

    # Mendeley Dataset Specification
    DATASET_URL: str = "https://data.mendeley.com/datasets/xc3j9v382w/1"
    DATASET_DOI: str = "10.17632/xc3j9v382w.1"

settings = Settings()
