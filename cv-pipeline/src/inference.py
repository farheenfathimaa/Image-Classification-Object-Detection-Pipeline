"""
Inference script running local image classification using the exported ResNet-50 ONNX model
and showing results in a clean table format.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms


def preprocess_image(image_path):
    """
    Loads and preprocesses an image using the standard ResNet-50 validation transforms.
    
    Args:
        image_path: Path to the local image file.
    Returns:
        numpy.ndarray: Preprocessed image tensor with shape (1, 3, 224, 224).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Standard ImageNet normalization and validation crops
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open and pre-process
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    # Add batch dimension and convert to numpy array
    return tensor.unsqueeze(0).numpy()


def softmax(x):
    """
    Computes softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def print_table(results):
    """
    Prints results in a clean, human-readable ASCII table.
    
    Args:
        results: List of tuples (rank, class_name, confidence)
    """
    header = f"| {'Rank':^4} | {'Pet Class / Breed':<30} | {'Confidence':^12} |"
    divider = "+" + "-" * 6 + "+" + "-" * 32 + "+" + "-" * 14 + "+"
    
    print(divider)
    print(header)
    print(divider)
    for rank, breed, confidence in results:
        print(f"| {rank:^4} | {breed:<30} | {confidence:>10.2f}% |")
    print(divider)


def main():
    """
    Main function to parse arguments, run ONNX inference, and display results.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_model = os.path.join(project_root, "models", "resnet50_pets.onnx")
    default_classes = os.path.join(project_root, "models", "class_names.json")

    parser = argparse.ArgumentParser(description="ONNX-based Image Classification CLI Inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default=default_model, help="Path to ONNX model.")
    parser.add_argument("--classes-json", type=str, default=default_classes, help="Path to class names JSON file.")
    args = parser.parse_args()

    # Load classes mapping
    if not os.path.exists(args.classes_json):
        raise FileNotFoundError(
            f"Class names config file not found at: {args.classes_json}. "
            "Please run train.py first to generate the config."
        )
    with open(args.classes_json, "r") as f:
        class_names = json.load(f)

    # Load ONNX session
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"ONNX model file not found at: {args.model}. "
            "Please run export_onnx.py first to export it."
        )
    
    print(f"Loading ONNX model: {args.model}...")
    ort_session = ort.InferenceSession(args.model)

    # Preprocess image
    print(f"Preprocessing input image: {args.image}...")
    input_tensor = preprocess_image(args.image)

    # Run inference
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor}
    
    print("Running model inference...")
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]

    # Calculate probabilities
    probs = softmax(logits)[0]

    # Get top-3 predicted indices
    top3_idx = np.argsort(probs)[-3:][::-1]

    # Format predictions
    table_data = []
    for rank_idx, idx in enumerate(top3_idx, 1):
        class_name = class_names[idx]
        confidence = probs[idx] * 100
        table_data.append((rank_idx, class_name, confidence))

    # Print results
    print("\n--- Classification Results ---")
    print_table(table_data)


if __name__ == "__main__":
    main()
