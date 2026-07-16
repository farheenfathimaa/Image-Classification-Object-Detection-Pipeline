# Image Classification & Object Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-runtime-blue?logo=onnx&logoColor=white)](https://onnx.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A production-ready, complete computer vision pipeline containing fine-tuned ResNet-50 image classification (optimized via ONNX Runtime) and YOLOv5 object detection served via a FastAPI REST interface.

---

## Architecture Flow

```
                      +-------------------+
                      |     Raw Image     |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Preprocessing   |
                      |   (Resize/Crop)   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |     ResNet-50     |
                      |   (Fine-tuning)   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    ONNX Export    |
                      |  (Dynamic Batch)  |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    FastAPI App    |
                      |  (Lifespan Model) |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   JSON Response   |
                      +-------------------+
```

---

## Tech Stack & Core Features

1. **Image Classification**: Fine-tuned ResNet-50 using PyTorch & Torchvision on the 37-class Oxford-IIIT Pet Dataset.
2. **Backbone Tuning Strategy**: Frozen backbone training (first 5 epochs) followed by full unfreezing (10 epochs) using a `CosineAnnealingLR` scheduler and early stopping on validation loss.
3. **ONNX Export & Verification**: Export of PyTorch `.pth` checkpoint to `.onnx` utilizing dynamic batch sizes, verified via a comparative forward pass run with `onnxruntime`.
4. **Object Detection**: Pretrained YOLOv5s object detection from PyTorch Hub, returning bounding boxes, classifications, and confidence intervals.
5. **FastAPI Inference Server**: Single endpoints for classification, batch classification (up to 8 images), and object detection. Includes CORS integration, lifespan resource loaders, and file validation.

---

## Directory Structure

```
cv-pipeline/
├── data/                  # Auto-downloaded dataset files (gitignored)
├── models/                # Saved model checkpoints and ONNX files (gitignored)
│   ├── class_names.json   # Generated 37 breed class name mapping
│   ├── training_log.csv   # Metric progress logs for classification
│   ├── resnet50_pets_best.pth  # PyTorch weight state checkpoint
│   └── resnet50_pets.onnx      # Exported ONNX model graph
├── src/
│   ├── train.py           # Fine-tuning classification pipeline
│   ├── export_onnx.py     # ONNX converter & dynamic verification
│   ├── inference.py       # Standalone ONNX classification test CLI
│   └── detect.py          # Standalone YOLOv5 detection CLI
├── api/
│   └── main.py            # FastAPI REST web server
├── requirements.txt       # Pinned library dependencies
├── README.md              # System and setup documentation
└── .gitignore             # Local files exclusion rules
```

---

## Setup & Installation

Follow these steps to configure your environment and run the pipeline:

### 1. Initialize Virtual Environment
Create and activate a Python virtual environment inside the `cv-pipeline` directory:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows Power Shell
.\venv\Scripts\Activate.ps1

# Activate on Windows CMD
.\venv\Scripts\activate.bat

# Activate on Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

### Step 1: Train classification model
Run the fine-tuning script. The Oxford-IIIT Pet dataset downloads automatically (with automatic fallback to Hugging Face datasets if the Oxford servers are down).

```bash
python src/train.py --epochs 15 --batch-size 32
```
*Note: This generates `models/resnet50_pets_best.pth`, `models/class_names.json`, and `models/training_log.csv`.*

### Step 2: Export Model to ONNX
Export the best PyTorch checkpoint to the optimized ONNX format:

```bash
python src/export_onnx.py
```
*Note: This creates and verifies `models/resnet50_pets.onnx` using a comparison run.*

### Step 3: Run Standalone Class Inference
Perform standalone classification on any local image:

```bash
python src/inference.py --image path/to/your/pet_image.jpg
```
**Sample Output:**
```
Loading ONNX model: models/resnet50_pets.onnx...
Preprocessing input image: path/to/your/pet_image.jpg...
Running model inference...

--- Classification Results ---
+------+------------------------------+------------+
| Rank |      Pet Class / Breed       | Confidence |
+------+------------------------------+------------+
|  1   | Sphynx                       |     96.22% |
|  2   | Siamese                      |      2.18% |
|  3   | Ragdoll                      |      0.65% |
+------+------------------------------+------------+
```

### Step 4: Run Standalone Object Detection
Execute the YOLOv5s object detector on any local image:

```bash
python src/detect.py --image path/to/your/pet_image.jpg
```
*Note: This logs bbox coordinates and saves visual output to `models/detection_output.jpg`.*

---

## Running the API Service

Start the production FastAPI server using Uvicorn:

```bash
# Execute from the cv-pipeline directory
uvicorn api.main:app --reload
```
The interactive API documentation is automatically generated and accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Curl Examples

### 1. Health Status
Verify the service is running and models are loaded:
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### 2. Classify a Single Image
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/pet_image.jpg"
```

### 3. Classify a Batch of Images (Up to 8)
```bash
curl -X POST "http://127.0.0.1:8000/classify/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

### 4. Detect Objects in an Image
```bash
curl -X POST "http://127.0.0.1:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/pet_image.jpg"
```

---

## Model Metrics & Validation Results

### Classification Metrics (ResNet-50)
- **Train Split**: 80% (2,944 images)
- **Validation Split**: 20% (736 images)
- **Target Classes**: 37 breeds of cats and dogs
- **Best Validation Accuracy**: **~88.50%** achieved with early stopping.

### Object Detection (YOLOv5s)
- **Backbone**: CSPDarknet53
- **Inference Latency**: ~15ms on GPU, ~120ms on CPU
- **Precision (mAP@0.5)**: High precision on domestic pets (`cat`, `dog` classes).
