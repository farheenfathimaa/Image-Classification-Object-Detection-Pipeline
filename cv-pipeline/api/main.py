"""
FastAPI application serving ResNet-50 image classification (ONNX Runtime)
and YOLOv5 object detection (PyTorch Hub).
Includes CORS support, batch classification, robust error handling,
and lifespan-based model management.
"""

import io
import os
import json
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global dict to store models and metadata loaded during lifespan
models_dict = {}

# Standard ResNet-50 validation transforms
classification_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limits


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for FastAPI lifespan events.
    Loads classification (ONNX) and detection (YOLOv5) models on startup,
    and cleans them up on shutdown.
    """
    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(api_dir)
    onnx_path = os.path.join(project_root, "models", "resnet50_pets.onnx")
    classes_path = os.path.join(project_root, "models", "class_names.json")

    # 1. Load ResNet-50 ONNX model
    if os.path.exists(onnx_path) and os.path.exists(classes_path):
        print(f"Loading ResNet-50 ONNX model from: {onnx_path}...")
        try:
            models_dict["classification_session"] = ort.InferenceSession(onnx_path)
            with open(classes_path, "r") as f:
                models_dict["classes"] = json.load(f)
            print("ResNet-50 ONNX model loaded successfully.")
        except Exception as e:
            print(f"Error loading ResNet-50 ONNX: {e}")
            models_dict["classification_session"] = None
            models_dict["classes"] = None
    else:
        print("Warning: ResNet-50 ONNX model or class_names.json not found in models/.")
        print("Classification endpoints will fail until models are trained and exported.")
        models_dict["classification_session"] = None
        models_dict["classes"] = None

    # 2. Load YOLOv5s model from PyTorch Hub
    print("Loading YOLOv5s model from PyTorch Hub...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        models_dict["yolo_model"] = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, device=device)
        print("YOLOv5s model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv5s model from PyTorch Hub: {e}")
        models_dict["yolo_model"] = None

    yield
    # Shutdown logic: release loaded model references
    models_dict.clear()
    print("Cleaned up loaded models.")


# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="Image Classification & Object Detection Pipeline API",
    description="A FastAPI server running ResNet-50 Image Classification and YOLOv5 Object Detection.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def validate_and_read_image(file: UploadFile) -> bytes:
    """
    Validates that the uploaded file is an image, is not empty, and stays within size limits.
    
    Args:
        file: Uploaded file from the request.
    Returns:
        bytes: Raw image file contents.
    """
    # 1. Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not a valid image. Content-Type must start with 'image/'."
        )

    # 2. Read contents
    contents = await file.read()

    # 3. Check if empty
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{file.filename}' is empty."
        )

    # 4. Check file size limits
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' size ({len(contents) / (1024*1024):.2f}MB) exceeds the 10MB limit."
        )

    # Reset file read cursor so other readers can read it if needed
    await file.seek(0)
    return contents


def run_classification_inference(image_bytes: bytes) -> list:
    """
    Performs preprocessing and executes ResNet-50 inference via ONNX Runtime.
    
    Args:
        image_bytes: Raw binary image contents.
    Returns:
        list: Top-3 classification predictions as dicts.
    """
    session = models_dict.get("classification_session")
    classes = models_dict.get("classes")

    if not session or not classes:
        raise HTTPException(
            status_code=503,
            detail="Classification model is currently unavailable. Please train and export the model first."
        )

    # Preprocess image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = classification_transform(img).unsqueeze(0).numpy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Run inference
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: input_tensor})
    logits = ort_outputs[0]

    # Compute softmax probabilities
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = (e_x / np.sum(e_x, axis=1, keepdims=True))[0]

    # Get top-3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    
    predictions = []
    for idx in top3_indices:
        predictions.append({
            "class": classes[idx],
            "confidence": float(probs[idx])
        })
    return predictions


@app.get("/health")
def health_check():
    """
    API Health status check endpoint.
    """
    classification_available = models_dict.get("classification_session") is not None
    detection_available = models_dict.get("yolo_model") is not None
    return {
        "status": "ok",
        "models": {
            "resnet50-oxford-pets": "loaded" if classification_available else "not_loaded",
            "yolov5s-object-detection": "loaded" if detection_available else "not_loaded"
        }
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Endpoint for single image classification.
    """
    # Validate and read upload
    image_bytes = await validate_and_read_image(file)
    
    # Run ResNet-50 ONNX classification
    predictions = run_classification_inference(image_bytes)
    
    return {
        "filename": file.filename,
        "predictions": predictions
    }


@app.post("/classify/batch")
async def classify_images_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint for batch image classification (supports up to 8 images).
    """
    # Enforce batch size limits
    if len(files) > 8:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum limit. A maximum of 8 files can be classified at once."
        )
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded for batch classification."
        )

    batch_results = []
    
    # Validate and process each file in the batch
    for file in files:
        try:
            image_bytes = await validate_and_read_image(file)
            predictions = run_classification_inference(image_bytes)
            batch_results.append({
                "filename": file.filename,
                "status": "success",
                "predictions": predictions
            })
        except HTTPException as he:
            batch_results.append({
                "filename": file.filename,
                "status": "failed",
                "error": he.detail
            })
        except Exception as e:
            batch_results.append({
                "filename": file.filename,
                "status": "failed",
                "error": f"Internal error: {str(e)}"
            })

    return {"results": batch_results}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for single image object detection using YOLOv5.
    """
    yolo_model = models_dict.get("yolo_model")
    if not yolo_model:
        raise HTTPException(
            status_code=503,
            detail="Object detection model is currently unavailable. Please verify YOLOv5 status."
        )

    # Validate and read upload
    image_bytes = await validate_and_read_image(file)

    # Run YOLOv5s object detection
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = yolo_model(img)
        df = results.pandas().xyxy[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process object detection: {e}")

    # Build response structure
    detections = []
    for _, row in df.iterrows():
        detections.append({
            "class": row["name"],
            "confidence": float(row["confidence"]),
            "box": {
                "xmin": float(row["xmin"]),
                "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]),
                "ymax": float(row["ymax"])
            }
        })

    return {
        "filename": file.filename,
        "detections": detections
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
