"""
Small Gradio UI: upload an MRI image → ResNet18 real/fake prediction.

Run from ai/:
  python scripts/demo_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_SIZE, device, ensure_ai_cwd

LABELS = ["real", "fake"]
CHECKPOINT = Path("models/best_model.pt")

_tfm = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model: nn.Module | None = None


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    ensure_ai_cwd()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT.resolve()}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device())
    model.eval()
    _model = model
    return _model


@torch.no_grad()
def predict(image: Image.Image | None):
    if image is None:
        return None, "Upload an image first."
    model = get_model()
    img = image.convert("RGB")
    x = _tfm(img).unsqueeze(0).to(device())
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    scores = {LABELS[i]: float(probs[i]) for i in range(2)}
    pred = LABELS[int(probs[1] >= probs[0])]
    conf = scores[pred]
    summary = f"**{pred.upper()}** — confidence {conf:.1%}"
    return scores, summary


def main():
    ensure_ai_cwd()
    print("Loading model…", flush=True)
    get_model()  # fail fast if checkpoint missing
    print(f"Model ready on {device()}", flush=True)
    with gr.Blocks(title="MedShield AI — Real vs Fake") as demo:
        gr.Markdown(
            "# MedShield AI\n"
            "Upload a brain MRI scan. The ResNet18 model predicts **real** or **fake**."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="MRI image", height=360)
            with gr.Column():
                out_label = gr.Label(label="Probabilities", num_top_classes=2)
                out_text = gr.Markdown()
        btn = gr.Button("Predict", variant="primary")
        btn.click(predict, inputs=inp, outputs=[out_label, out_text])
        inp.change(predict, inputs=inp, outputs=[out_label, out_text])
    demo.queue(max_size=8)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
