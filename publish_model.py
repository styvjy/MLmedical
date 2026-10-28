"""
Model Publishing & Export Utility
Exports trained PyTorch weights to:
1. ONNX (Open Neural Network Exchange) for cross-platform, edge, and browser inference.
2. TorchScript (.pt) for high-performance C++ / PyTorch serving without Python dependency.
3. Hugging Face Hub package with complete Model Card.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.config import BEST_MODEL_PATH, MODELS_DIR, IMAGE_SIZE, NUM_CLASSES
from src.model import build_mobilenetv2, load_checkpoint

class ProbabilitiesWrapper(nn.Module):
    """Wraps model to output calibrated Softmax probabilities directly in exported artifacts."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return self.softmax(logits)

def export_to_onnx(model_path: Path, output_path: Path) -> bool:
    """Exports model to ONNX format with dynamic batch sizing."""
    print(f"\n[INFO] Exporting PyTorch model to ONNX: {output_path}...")
    try:
        model = build_mobilenetv2(num_classes=NUM_CLASSES, pretrained=False)
        load_checkpoint(model, model_path, device=torch.device("cpu"))
        wrapped_model = ProbabilitiesWrapper(model)
        wrapped_model.eval()

        dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], requires_grad=False)

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["probabilities"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "probabilities": {0: "batch_size"}
            }
        )
        print(f"[SUCCESS] ONNX model successfully saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False

def export_to_torchscript(model_path: Path, output_path: Path) -> bool:
    """Exports model to traced TorchScript for C++/server deployments."""
    print(f"\n[INFO] Exporting model to TorchScript: {output_path}...")
    try:
        model = build_mobilenetv2(num_classes=NUM_CLASSES, pretrained=False)
        load_checkpoint(model, model_path, device=torch.device("cpu"))
        wrapped_model = ProbabilitiesWrapper(model)
        wrapped_model.eval()

        dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        traced = torch.jit.trace(wrapped_model, dummy_input)
        traced.save(str(output_path))
        print(f"[SUCCESS] TorchScript model successfully saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] TorchScript export failed: {e}")
        return False

def generate_hf_model_card(output_dir: Path):
    """Generates a professional Hugging Face Model Card README.md."""
    card_content = """---
language:
- en
license: mit
library_name: pytorch
tags:
- medical
- image-classification
- chest-xray
- pneumonia
- mobilenetv2
- computer-vision
metrics:
- accuracy
- precision
- recall
- f1
pipeline_tag: image-classification
---

# Chest X-Ray Pneumonia Image Classifier (MobileNetV2)

A convolutional neural network model utilizing MobileNetV2 transfer learning to classify anterior-posterior chest radiographs into **NORMAL** and **PNEUMONIA**.

## Model Details
- **Architecture**: MobileNetV2 (1280-dim feature extractor + Custom Regularized Head)
- **Input Resolution**: 224 x 224 x 3
- **Normalization**: ImageNet Standard (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Classes**: `0: NORMAL`, `1: PNEUMONIA`

## Intended Use
Research and educational machine learning demonstrations. **Not for clinical diagnosis.**

## Quick Inference Example (PyTorch)
```python
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("chest_xray.jpeg").convert("RGB")
tensor = transform(img).unsqueeze(0)
```
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"[SUCCESS] Hugging Face Model Card written to: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Publish & Export Chest X-Ray Classifier")
    parser.add_argument("--format", choices=["onnx", "torchscript", "all"], default="all")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model weights not found at: {model_path}")
        print("Please train the model first by running: python -m src.train")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["onnx", "all"]:
        onnx_path = MODELS_DIR / "mobilenetv2_pneumonia.onnx"
        export_to_onnx(model_path, onnx_path)

    if args.format in ["torchscript", "all"]:
        ts_path = MODELS_DIR / "mobilenetv2_pneumonia_traced.pt"
        export_to_torchscript(model_path, ts_path)

    generate_hf_model_card(MODELS_DIR)

if __name__ == "__main__":
    main()
