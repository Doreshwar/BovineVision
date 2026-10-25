"""
BovineVision AI - Cattle Breed Classification Service
Handles model loading, image preprocessing, GPU inference,
top-3 confidence extraction, and threshold rejection logic.
Completely separate from RAG.
"""

import os
import json
import io
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from backend.app.models.cnn import BovineVisionCNN


class BreedClassifierService:
    def __init__(
        self,
        checkpoint_path: str = "G:/Projektz/backend/saved_models/best_model.pth",
        classes_path: str = "G:/Projektz/backend/saved_models/classes.json",
        default_threshold: float = 0.50
    ):
        self.checkpoint_path = checkpoint_path
        self.classes_path = classes_path
        self.default_threshold = default_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BovineVisionCNN] = None
        self.classes: List[str] = []
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        """Loads trained weights onto the target compute device."""
        if not os.path.exists(self.checkpoint_path):
            print(f"[Classifier] Warning: Model checkpoint not found at {self.checkpoint_path}")
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            self.classes = checkpoint.get("classes", [])
            if not self.classes and os.path.exists(self.classes_path):
                with open(self.classes_path, "r") as f:
                    self.classes = json.load(f)

            num_classes = len(self.classes)
            config = checkpoint.get("config", {})

            self.model = BovineVisionCNN(
                in_channels=config.get("in_channels", 3),
                num_classes=num_classes,
                stem_channels=config.get("stem_channels", 32),
                stage_channels=config.get("stage_channels", [32, 64, 128, 256]),
                blocks_per_stage=config.get("blocks_per_stage", [2, 2, 2, 2]),
                strides=config.get("strides", [1, 2, 2, 2]),
                use_se=config.get("use_se", True),
                dropout_rate=config.get("dropout_rate", 0.25),
                classifier_hidden=config.get("classifier_hidden", 256)
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"[Classifier] Model successfully loaded on {self.device} with {num_classes} classes: {self.classes}")
        except Exception as e:
            print(f"[Classifier] Error loading model: {e}")

    def predict(
        self,
        image_data: bytes,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes model prediction on the provided image bytes.
        Returns top 3 predictions with confidence percentages.
        Applies minimum threshold logic (default 50%).
        If top prediction < threshold, returns rejection with message:
        'Could not detect, try another picture'
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError("Classifier model weights are not loaded.")

        applied_threshold = threshold if threshold is not None else self.default_threshold

        # Load and preprocess image
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        # Retrieve top 3 predictions
        top_k = min(3, len(self.classes))
        confidences, indices = torch.topk(probabilities, k=top_k)

        top_3 = []
        for rank, (conf, idx) in enumerate(zip(confidences.cpu().numpy(), indices.cpu().numpy()), start=1):
            breed_name = self.classes[idx]
            conf_val = float(conf)
            top_3.append({
                "rank": rank,
                "breed": breed_name,
                "confidence": round(conf_val, 4),
                "confidence_percentage": round(conf_val * 100, 1)
            })

        best_candidate = top_3[0]
        # Minimum threshold evaluation
        if best_candidate["confidence"] < applied_threshold:
            is_accepted = False
            rejection_message = "Could not detect, try another picture"
            can_trigger_rag = False
            top_prediction = None
        else:
            is_accepted = True
            rejection_message = None
            can_trigger_rag = True
            top_prediction = best_candidate

        return {
            "is_accepted": is_accepted,
            "rejection_message": rejection_message,
            "can_trigger_rag": can_trigger_rag,
            "threshold_applied": applied_threshold,
            "threshold_percentage": round(applied_threshold * 100, 1),
            "top_prediction": top_prediction,
            "top_3_predictions": top_3,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
