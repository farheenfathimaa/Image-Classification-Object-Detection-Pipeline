"""
Object detection script using YOLOv5s loaded from PyTorch Hub.
Draws bounding boxes and labels on detected objects and saves the visual result.
"""

import os
import argparse
import torch
from PIL import Image


def run_detection(image_path, model_name="yolov5s", device="cpu"):
    """
    Loads YOLOv5 model from PyTorch Hub, performs inference on the image,
    and returns the detections and the rendered image.
    
    Args:
        image_path: Path to the local input image file.
        model_name: YOLOv5 model variant (default: yolov5s).
        device: Device to run inference on (cpu or cuda).
    Returns:
        df: Pandas DataFrame containing the bounding box coordinates, confidence and class names.
        rendered_image: PIL Image with bounding boxes and labels drawn on it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    print(f"Loading {model_name} from PyTorch Hub (ultralytics/yolov5)...")
    # Load model from ultralytics repository
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, device=device)
    
    print(f"Running detection on: {image_path}...")
    # Load PIL image and run inference
    img = Image.open(image_path).convert("RGB")
    results = model(img)

    # Fetch pandas dataframe containing coordinates
    df = results.pandas().xyxy[0]

    # Render bounding boxes onto the image
    results.render()  # This updates results.ims with the rendered RGB numpy arrays
    rendered_np = results.ims[0]
    rendered_image = Image.fromarray(rendered_np)

    return df, rendered_image


def main():
    """
    Main function to parse CLI arguments, run detection, print summary and save output.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_output = os.path.join(project_root, "models", "detection_output.jpg")

    parser = argparse.ArgumentParser(description="Object detection script using YOLOv5s.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save the output image with boxes.")
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLOv5 model to load.")
    args = parser.parse_args()

    # Determine execution device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running object detection on device: {device}")

    # Run detection
    try:
        detections_df, output_img = run_detection(args.image, model_name=args.model, device=device)
    except Exception as e:
        print(f"Error during detection run: {e}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save the output image
    output_img.save(args.output)
    print(f"Successfully saved annotated output image to: {args.output}")

    # Print detection summary
    print("\n--- Detection Summary ---")
    if len(detections_df) == 0:
        print("No objects detected in the image.")
    else:
        for idx, row in detections_df.iterrows():
            print(
                f"Class: {row['name']:<12} | "
                f"Confidence: {row['confidence']:.2f} | "
                f"Bounding Box: [{row['xmin']:.1f}, {row['ymin']:.1f}, {row['xmax']:.1f}, {row['ymax']:.1f}]"
            )
    print("-------------------------\n")


if __name__ == "__main__":
    main()
