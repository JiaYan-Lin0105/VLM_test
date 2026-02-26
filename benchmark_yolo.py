from ultralytics import YOLO
import time
import os
import torch
import numpy as np

def benchmark_model(model_path, device, num_runs=100, warmup_runs=10):
    """
    Loads a model and benchmarks its inference speed.
    """
    if not os.path.exists(model_path):
        print(f"Skipping {model_path}: File not found.")
        return None

    print(f"\nBenchmarking {model_path} on {device}...")
    
    try:
        # Load model using Ultralytics interface
        # Ultralytics handles loading .pt, .onnx, .torchscript, and .engine automatically
        model = YOLO(model_path, task='detect')
        
        # Create a dummy image for warming up and testing
        # Use a standard size like 640x640
        dummy_input = "https://ultralytics.com/images/bus.jpg" # Using a sample URL or local file matches real-world use better
        
        # Warmup
        print(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            model.predict(dummy_input, verbose=False, device=device)
            
        # Benchmark
        print(f"Running benchmark ({num_runs} runs)...")
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            model.predict(dummy_input, verbose=False, device=device)
            end_time = time.time()
            times.append((end_time - start_time) * 1000) # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        print(f"Average Inference Time: {avg_time:.2f} ms ± {std_time:.2f} ms")
        print(f"Approximate FPS: {fps:.2f}")
        
        return {
            "Model": os.path.basename(model_path),
            "Format": model_path.split('.')[-1],
            "Avg Time (ms)": avg_time,
            "FPS": fps
        }

    except Exception as e:
        print(f"Failed to benchmark {model_path}: {e}")
        return None

if __name__ == "__main__":
    # Base model name
    base_name = "yolo11x"
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {'GPU' if device == 0 else 'CPU'}")

    # List of models to test
    # Note: Ultralytics export names usually follow these conventions
    models_to_test = [
        f"{base_name}.pt",           # PyTorch
        f"{base_name}.onnx",         # ONNX
        f"{base_name}.torchscript",  # TorchScript 
        f"{base_name}.engine"        # TensorRT (creates a folder or file depending on version, often .engine)
    ]

    results = []

    for model_file in models_to_test:
        res = benchmark_model(model_file, device)
        if res:
            results.append(res)

    # Print Summary Table
    if results:
        print("\n" + "="*60)
        print(f"{'Model Format':<15} | {'Avg Time (ms)':<15} | {'FPS':<10} | {'Speedup (vs PT)':<15}")
        print("-" * 60)
        
        # Find baseline (PT)
        baseline = next((r for r in results if r['Format'] == 'pt'), None)
        baseline_time = baseline['Avg Time (ms)'] if baseline else None

        for res in results:
            speedup_str = "-"
            if baseline_time and res['Format'] != 'pt':
                speedup = baseline_time / res['Avg Time (ms)']
                speedup_str = f"{speedup:.2f}x"
            elif res['Format'] == 'pt':
                speedup_str = "1.00x"
                
            print(f"{res['Format']:<15} | {res['Avg Time (ms)']:<15.2f} | {res['FPS']:<10.2f} | {speedup_str:<15}")
        print("="*60)
    else:
        print("\nNo models benchmarked.")
