"""
AgroVision AI — Phase 0: Verification Script
=============================================
Verifies exported models are correct by:
  1. Comparing PyTorch vs ONNX FP32 output
  2. Comparing PyTorch vs ONNX INT8 (quantized) output
  3. Benchmarking inference speed of all formats
  4. Running a sample prediction with class name mapping
"""

import os
import sys
import json
import time
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.factory import create_model
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def load_class_names(path="shared/class_names.json"):
    with open(path, 'r') as f:
        return json.load(f)


def preprocess_image(image_path, image_size=224):
    """Preprocess a single image — same as evaluation transforms."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def benchmark_pytorch(model, dummy_input, n_runs=50):
    """Benchmark PyTorch CPU inference speed."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


def benchmark_onnx(onnx_path, dummy_input_np, n_runs=50):
    """Benchmark ONNX Runtime CPU inference speed."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input_np})

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input_np})
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


def verify_model(config_path, test_image_path=None):
    """Run full verification pipeline."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    save_dir = config['training']['save_dir']
    checkpoint_name = config['training'].get('checkpoint_name', 'best_model.pth')
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    print(f"\n{'='*60}")
    print(f"  AgroVision AI — Verification: {model_name}")
    print(f"{'='*60}")

    # Load model
    model = create_model(config)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    )
    model.eval()

    # Prepare input
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n[i] Using test image: {test_image_path}")
        dummy_input = preprocess_image(test_image_path)
    else:
        print(f"\n[i] Using random dummy input (no test image provided)")
        dummy_input = torch.randn(1, 3, 224, 224)

    dummy_input_np = dummy_input.numpy()

    # PyTorch baseline
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()
    pt_probs = np.exp(pt_output) / np.sum(np.exp(pt_output))  # softmax

    # Load class names
    class_names = load_class_names()

    print(f"\n--- PyTorch Prediction ---")
    top5_idx = np.argsort(pt_probs[0])[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  #{rank}: {class_names[idx]} ({pt_probs[0][idx]*100:.2f}%)")

    # Check ONNX files
    onnx_fp32 = os.path.join(save_dir, "best_model.onnx")
    onnx_int8 = os.path.join(save_dir, "best_model_quantized.onnx")

    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[✗] onnxruntime not installed. Skipping ONNX verification.")
        return

    # Verify ONNX FP32
    if os.path.exists(onnx_fp32):
        print(f"\n--- ONNX FP32 Verification ---")
        session = ort.InferenceSession(onnx_fp32)
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy_input_np})[0]

        max_diff = np.max(np.abs(pt_output - ort_output))
        print(f"  Max diff (PyTorch vs ONNX FP32): {max_diff:.8f}")
        print(f"  Status: {'[PASSED]' if max_diff < 1e-4 else '[CHECK]'}")
    else:
        print(f"\n[!] ONNX FP32 not found: {onnx_fp32}")

    # Verify ONNX INT8
    if os.path.exists(onnx_int8):
        print(f"\n--- ONNX INT8 Verification ---")
        session_q = ort.InferenceSession(onnx_int8)
        input_name_q = session_q.get_inputs()[0].name
        ort_q_output = session_q.run(None, {input_name_q: dummy_input_np})[0]

        max_diff_q = np.max(np.abs(pt_output - ort_q_output))
        ort_q_probs = np.exp(ort_q_output) / np.sum(np.exp(ort_q_output))
        pt_top1 = np.argmax(pt_probs[0])
        q_top1 = np.argmax(ort_q_probs[0])

        print(f"  Max diff (PyTorch vs ONNX INT8): {max_diff_q:.6f}")
        print(f"  Top-1 match: {'[YES]' if pt_top1 == q_top1 else '[NO]'}")
        print(f"  PyTorch top-1: {class_names[pt_top1]}")
        print(f"  ONNX INT8 top-1: {class_names[q_top1]}")
    else:
        print(f"\n[!] ONNX INT8 not found: {onnx_int8}")

    # Benchmark
    print(f"\n--- Inference Speed Benchmark (CPU, 50 runs) ---")
    print(f"  {'Format':<30} {'Mean (ms)':>10} {'Std (ms)':>10}")
    print(f"  {'-'*50}")

    pt_mean, pt_std = benchmark_pytorch(model, dummy_input)
    print(f"  {'PyTorch (FP32)':<30} {pt_mean:>10.2f} {pt_std:>10.2f}")

    if os.path.exists(onnx_fp32):
        ort_mean, ort_std = benchmark_onnx(onnx_fp32, dummy_input_np)
        print(f"  {'ONNX Runtime (FP32)':<30} {ort_mean:>10.2f} {ort_std:>10.2f}")

    if os.path.exists(onnx_int8):
        ort_q_mean, ort_q_std = benchmark_onnx(onnx_int8, dummy_input_np)
        print(f"  {'ONNX Runtime (INT8)':<30} {ort_q_mean:>10.2f} {ort_q_std:>10.2f}")

    # File sizes
    print(f"\n--- File Sizes ---")
    for name, path in [
        ("Original .pth", checkpoint_path),
        ("ONNX FP32", onnx_fp32),
        ("ONNX INT8 (Edge)", onnx_int8),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name:<30} {size:>8.2f} MB")

    print(f"\n{'='*60}")
    print(f"  Verification Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Verify exported models")
    parser.add_argument("--config", type=str, default="configs/mobilenet_v3.yaml")
    parser.add_argument("--image", type=str, default="test_image.jpg",
                        help="Path to test image for sample prediction")
    args = parser.parse_args()

    verify_model(args.config, args.image)
