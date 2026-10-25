"""
BovineVision AI - REST API Endpoints
Exposes classification and RAG services via clean, modular FastAPI endpoints.
Strictly separates classification and RAG.
"""

import os
import json
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.app.services.classifier import BreedClassifierService
from backend.app.services.rag import RAGService

router = APIRouter(prefix="/api")

# Services
classifier_service = BreedClassifierService()
rag_service = RAGService()


class QuestionRequest(BaseModel):
    breed_name: str
    question: str


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "BovineVision AI API",
        "device": str(classifier_service.device),
        "device_name": classifier_service.predict.__defaults__ if False else (
            classifier_service.device.type
        ),
        "classes_supported": classifier_service.classes,
        "rag_breeds_indexed": list(rag_service.breed_dossiers.keys())
    }


@router.post("/classify")
def classify_image(
    file: UploadFile = File(...),
    threshold: float = 0.50
):
    """
    Classifies an uploaded cattle image using the custom BovineVisionCNN.
    Returns Top 3 predictions with confidence percentages.
    Applies minimum confidence threshold logic.
    If none cross threshold, rejects with: 'Could not detect, try another picture'
    """
    content_type = file.content_type or ""
    if not (content_type.startswith("image/") or file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG, PNG).")

    contents = file.file.read()


    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = classifier_service.predict(contents, threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/rag/breed/{breed_name}")
def get_breed_knowledge(
    breed_name: str,
    is_rejected: bool = False
):
    """
    RAG Endpoint: Retrieves verified breed knowledge dossier.
    Enforces strict decoupling: Rejected predictions cannot trigger RAG responses.
    """
    if is_rejected is True:
        raise HTTPException(
            status_code=403,
            detail="RAG responses are prohibited for unverified/rejected predictions."
        )


    try:
        dossier = rag_service.get_breed_dossier(breed_name, is_rejected=False)
        return dossier
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Breed '{breed_name}' not found in knowledge repository.")
    except ValueError as ve:
        raise HTTPException(status_code=403, detail=str(ve))


@router.post("/rag/ask")
def ask_breed_rag(request: QuestionRequest):
    """
    Interactive RAG QA: Answers questions regarding a verified cattle breed
    grounded strictly in the ICAR-NBAGR verified knowledge base.
    """
    breed = request.breed_name.strip()
    question = request.question.strip()
    if not breed or not question:
        raise HTTPException(status_code=400, detail="Both 'breed_name' and 'question' are required.")

    result = rag_service.answer_question(breed, question, is_rejected=False)
    return result


@router.get("/model/info")
def get_model_info():
    """
    Returns custom CNN architecture parameters, hardware compute details,
    training metrics, and evaluation results.
    """
    summary = classifier_service.model.get_architecture_summary() if classifier_service.model else {}
    metrics_path = "G:/Projektz/backend/saved_models/metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return {
        "model_architecture": summary,
        "classes": classifier_service.classes,
        "device": str(classifier_service.device),
        "metrics": metrics
    }


@router.get("/model/confusion-matrix")
def get_confusion_matrix_image():
    cm_path = "G:/Projektz/backend/saved_models/confusion_matrix.png"
    if not os.path.exists(cm_path):
        raise HTTPException(status_code=404, detail="Confusion matrix plot not generated yet.")
    return FileResponse(cm_path, media_type="image/png")


@router.get("/model/training-history")
def get_training_history_image():
    plot_path = "G:/Projektz/backend/saved_models/training_history.png"
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Training history plot not generated yet.")
    return FileResponse(plot_path, media_type="image/png")


@router.get("/samples")
def get_test_samples():
    """Returns accessible sample images from test split for 1-click evaluation."""
    test_dir = "G:/Projektz/data/test"
    samples = []
    if os.path.exists(test_dir):
        for breed in sorted(os.listdir(test_dir)):
            b_dir = os.path.join(test_dir, breed)
            if os.path.isdir(b_dir):
                files = [f for f in os.listdir(b_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    samples.append({
                        "breed": breed,
                        "filename": files[0],
                        "sample_url": f"/api/samples/{breed}/{files[0]}"
                    })
    return {"samples": samples}


@router.get("/samples/{breed}/{filename}")
def serve_sample_image(breed: str, filename: str):
    p = os.path.join("G:/Projektz/data/test", breed, filename)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(p, media_type="image/jpeg")
