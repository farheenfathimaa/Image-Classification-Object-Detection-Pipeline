"""
Script to export the fine-tuned ResNet-50 PyTorch model checkpoint to ONNX format
and verify the export using ONNX Runtime.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import onnxruntime as ort


def load_model(checkpoint_path, num_classes, device):
    """
    Loads the ResNet-50 model structure and maps the trained checkpoint weights.
    """
    print(f"Initializing ResNet-50 structure with {num_classes} classes...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    print(f"Loading weights from checkpoint: {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    """
    Main function to parse arguments, export model to ONNX, and run verification.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_checkpoint = os.path.join(project_root, "models", "resnet50_pets_best.pth")
    default_output = os.path.join(project_root, "models", "resnet50_pets.onnx")
    default_classes = os.path.join(project_root, "models", "class_names.json")

    parser = argparse.ArgumentParser(description="Export PyTorch ResNet-50 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to PyTorch model checkpoint.")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output ONNX file.")
    parser.add_argument("--classes-json", type=str, default=default_classes, help="Path to class names JSON file.")
    args = parser.parse_args()

    # Determine device (CPU for export is recommended and standard)
    device = torch.device("cpu")

    # Load classes and count
    if not os.path.exists(args.classes_json):
        raise FileNotFoundError(
            f"Class names config file not found at: {args.classes_json}. "
            "Please run train.py first to generate the model and its metadata."
        )
        
    with open(args.classes_json, "r") as f:
        classes = json.load(f)
    num_classes = len(classes)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {args.checkpoint}. Please train the model first."
        )

    # Load the PyTorch model
    model = load_model(args.checkpoint, num_classes, device)

    # Generate dummy input: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Exporting model to ONNX format: {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print("Model successfully exported to ONNX.")

    # 3. Verify with ONNX Runtime
    print("Verifying ONNX export with ONNX Runtime...")
    ort_session = ort.InferenceSession(args.output)
    
    # Prepare dummy input for ONNX Runtime
    dummy_np = dummy_input.numpy()
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_np}
    
    # Run dynamic batch inference pass
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_logits = ort_outputs[0]
    
    # Run reference forward pass in PyTorch
    with torch.no_grad():
        pytorch_logits = model(dummy_input).numpy()
        
    # Check outputs matching shape and closeness
    assert ort_logits.shape == pytorch_logits.shape, (
        f"Shape mismatch: PyTorch shape {pytorch_logits.shape} vs ONNX shape {ort_logits.shape}"
    )
    
    np.testing.assert_allclose(
        pytorch_logits,
        ort_logits,
        rtol=1e-03,
        atol=1e-05,
        err_msg="PyTorch and ONNX Runtime predictions differ significantly."
    )
    
    print("ONNX export verified successfully")


if __name__ == "__main__":
    main()
