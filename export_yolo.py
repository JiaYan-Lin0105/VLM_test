from ultralytics import YOLO
import torch
import os

def export_model():
    model_path = "yolo11x.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in the current directory.")
        return

    print(f"Loading model: {model_path}...")
    try:
        # Load the YOLO model
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'ultralytics' is installed: pip install ultralytics")
        return

    # 1. Export to ONNX
    print("\nStarting ONNX export...")
    try:
        # dynamic=True allows for different batch sizes and image sizes
        model.export(format="onnx", dynamic=True)
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # 2. Export to TorchScript
    print("\nStarting TorchScript export...")
    try:
        model.export(format="torchscript")
        print("TorchScript export successful.")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # 3. Export to TensorRT
    print("\nStarting TensorRT export...")
    if torch.cuda.is_available():
        try:
            # device=0 ensures it runs on the first GPU
            # This might require 'tensorrt' python package installed
            model.export(format="engine", device=0)
            print("TensorRT export successful.")
        except Exception as e:
            print(f"TensorRT export failed: {e}")
            print("Note: TensorRT export requires a GPU and valid CUDA/TensorRT installation.")
    else:
        print("Skipping TensorRT export: No CUDA GPU detected.")

if __name__ == "__main__":
    export_model()
