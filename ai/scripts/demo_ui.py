"""
Small Gradio UI: upload an MRI image → real/fake prediction.

With a localizer checkpoint it also shows a heatmap of where the image was edited.

Run from ai/:
  python scripts/demo_ui.py
  python scripts/demo_ui.py --checkpoint models/localizer.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_SIZE, device, ensure_ai_cwd
from localizer import ARCH as LOC_ARCH
from localizer import LOC_SIZE, Localizer, image_scores, preprocess

LABELS = ["real", "fake"]
CHECKPOINT = Path("models/best_model.pt")
LOC_METRICS = Path("metrics/metrics_localizer.json")

_tfm = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_state: dict = {}


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_model() -> dict:
    if _state:
        return _state
    ensure_ai_cwd()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT.resolve()}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    is_loc = ckpt.get("arch") == LOC_ARCH
    model = Localizer() if is_loc else build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device()).eval()
    score, thr = "combined", 0.5
    if is_loc and LOC_METRICS.exists():
        report = json.loads(LOC_METRICS.read_text(encoding="utf-8"))
        if report.get("checkpoint") and Path(report["checkpoint"]) == CHECKPOINT:
            score, thr = report["image_score"], float(report["threshold"])
    _state.update(model=model, is_loc=is_loc, score=score, threshold=thr)
    return _state


def overlay(img: Image.Image, heat: np.ndarray) -> Image.Image:
    import matplotlib

    base = img.convert("RGB")
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat_img) / 255.0)[..., :3]
    alpha = (np.asarray(heat_img, np.float32) / 255.0)[..., None] * 0.6
    out = np.asarray(base, np.float32) / 255.0 * (1 - alpha) + colored * alpha
    return Image.fromarray((out * 255).astype(np.uint8))


@torch.no_grad()
def predict(image: Image.Image | None):
    if image is None:
        return None, "Upload an image first.", None
    st = get_model()
    model = st["model"]
    if st["is_loc"]:
        x = preprocess(image, LOC_SIZE)[None].to(device())
        mask_logits, cls_logit = model(x)
        p_fake = float(image_scores(mask_logits, cls_logit)[st["score"]][0])
        heat = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        thr = st["threshold"]
        pred = "fake" if p_fake >= thr else "real"
        scores = {"fake": p_fake, "real": 1 - p_fake}
        summary = (
            f"**{pred.upper()}** — fake score {p_fake:.1%} (threshold {thr:.1%})\n\n"
            f"Most suspicious region covers {float((heat > 0.5).mean()):.1%} of the image."
        )
        return scores, summary, overlay(image, heat)

    x = _tfm(image.convert("RGB")).unsqueeze(0).to(device())
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    scores = {LABELS[i]: float(probs[i]) for i in range(2)}
    pred = LABELS[int(probs[1] >= probs[0])]
    return scores, f"**{pred.upper()}** — confidence {scores[pred]:.1%}", None


def main():
    global CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    CHECKPOINT = Path(args.checkpoint)

    ensure_ai_cwd()
    print("Loading model…", flush=True)
    st = get_model()
    kind = "localizer" if st["is_loc"] else "classifier"
    print(f"Model ready on {device()}: {CHECKPOINT.as_posix()} ({kind})", flush=True)
    with gr.Blocks(title="MedShield AI — Real vs Fake") as demo:
        gr.Markdown(
            "# MedShield AI\n"
            "Upload a brain MRI scan. The model predicts **real** or **fake**"
            + (" and highlights the edited region." if st["is_loc"] else ".")
            + f"\n\nCheckpoint: `{CHECKPOINT.as_posix()}`"
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="MRI image", height=360)
            with gr.Column():
                out_label = gr.Label(label="Probabilities", num_top_classes=2)
                out_text = gr.Markdown()
            out_heat = gr.Image(type="pil", label="Suspected edit (heatmap)", height=360, visible=st["is_loc"])
        btn = gr.Button("Predict", variant="primary")
        outputs = [out_label, out_text, out_heat]
        btn.click(predict, inputs=inp, outputs=outputs)
        inp.change(predict, inputs=inp, outputs=outputs)
    demo.queue(max_size=8)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
