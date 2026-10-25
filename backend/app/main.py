"""
BovineVision AI - FastAPI Backend Application
Main application entry point configuring CORS, routes, static assets, and lifecycle events.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.endpoints import router as api_router

app = FastAPI(
    title="BovineVision AI API",
    description="High-precision custom PyTorch CNN and RAG system for Indian Cattle Breed Identification and Intelligence.",
    version="1.0.0"
)

# CORS Configuration for React TypeScript Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount saved models directory for static access if needed
saved_models_dir = "G:/Projektz/backend/saved_models"
if os.path.exists(saved_models_dir):
    app.mount("/static/models", StaticFiles(directory=saved_models_dir), name="models")


@app.get("/")
def root():
    return {
        "project": "BovineVision AI",
        "description": "Custom PyTorch CNN (trained from scratch) with decoupled RAG knowledge system",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
