"""
Parallel GAN+ViT Deepfake Detector — Inference Script
Usage:
    python test/parallel.py                        # prompts for path
    python test/parallel.py --input path/to/file   # non-interactive
"""

import sys
import os
import argparse

import torch
import numpy as np
import cv2

# ── resolve project root so imports work from any working directory ──────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.parallel_model import ParallelGANViT

# ── constants ────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'parallel', 'ParallelGANViT_best.pth')
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

FAKE_THRESHOLD = 0.5   # probability above which the sample is classified FAKE


# ── preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR numpy image (H×W×3) to a normalised tensor (1×3×224×224)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # 1×3×H×W, float32
    return tensor


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def extract_frames(path: str, max_frames: int = 20) -> list:
    """Extract up to max_frames uniformly-spaced frames from a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1000

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be extracted from: {path}")
    return frames


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(device: torch.device) -> ParallelGANViT:
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Train the model first with: python train.py"
        )
    model = ParallelGANViT(
        img_size=IMG_SIZE,
        gan_feature_dim=512,
        vit_embed_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1
    )
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_frame(model: ParallelGANViT, frame: np.ndarray,
                  device: torch.device) -> tuple:
    """Return (fake_probability, confidence_score) for a single frame."""
    tensor = preprocess_image(frame).to(device)
    out    = model(tensor)
    prob   = out['output'].squeeze().item()
    conf   = out['confidence'].squeeze().item()
    return prob, conf


def classify(prob: float) -> tuple:
    label = 'FAKE' if prob >= FAKE_THRESHOLD else 'REAL'
    pct   = prob * 100 if label == 'FAKE' else (1 - prob) * 100
    return label, pct


# ── main ──────────────────────────────────────────────────────────────────────
def run(input_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Parallel GAN+ViT Deepfake Detector]")
    print(f"Device    : {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("-" * 55)

    print("Loading model ...", end=' ', flush=True)
    model = load_model(device)
    print("done.")

    ext = os.path.splitext(input_path)[1].lower()

    if ext in IMAGE_EXTS:
        # ── single image ──────────────────────────────────────────────────────
        print(f"Input     : image  →  {input_path}")
        frame = load_image(input_path)
        prob, model_conf = predict_frame(model, frame, device)
        label, conf = classify(prob)

        print(f"\n{'='*55}")
        print(f"  Result            :  {'[!] ' if label == 'FAKE' else '[✓] '}{label}")
        print(f"  Confidence        :  {conf:.2f}%")
        print(f"  Model certainty   :  {model_conf * 100:.2f}%")
        print(f"  Raw score         :  {prob:.4f}  (threshold = {FAKE_THRESHOLD})")
        print(f"{'='*55}\n")

    elif ext in VIDEO_EXTS:
        # ── video (per-frame + aggregate) ────────────────────────────────────
        print(f"Input     : video  →  {input_path}")
        print("Extracting frames ...", end=' ', flush=True)
        frames = extract_frames(input_path, max_frames=20)
        print(f"{len(frames)} frames extracted.")

        probs       = []
        model_confs = []

        for i, frame in enumerate(frames, 1):
            p, mc = predict_frame(model, frame, device)
            probs.append(p)
            model_confs.append(mc)
            lbl, conf = classify(p)
            print(f"  Frame {i:>3}/{len(frames)}  →  {lbl}  ({conf:.1f}%)  "
                  f"[model certainty: {mc*100:.1f}%]")

        avg_prob = float(np.mean(probs))
        avg_conf = float(np.mean(model_confs))
        label, conf = classify(avg_prob)

        print(f"\n{'='*55}")
        print(f"  Aggregate Result  :  {'[!] ' if label == 'FAKE' else '[✓] '}{label}")
        print(f"  Avg Confidence    :  {conf:.2f}%")
        print(f"  Avg Model Cert.   :  {avg_conf * 100:.2f}%")
        print(f"  Avg Raw Score     :  {avg_prob:.4f}  (threshold = {FAKE_THRESHOLD})")
        print(f"  Frames analysed   :  {len(frames)}")
        print(f"  Fake frame count  :  {sum(1 for p in probs if p >= FAKE_THRESHOLD)}")
        print(f"{'='*55}\n")

    else:
        print(f"Unsupported file type: '{ext}'")
        print(f"Supported image formats : {IMAGE_EXTS}")
        print(f"Supported video formats : {VIDEO_EXTS}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel GAN+ViT deepfake detector.")
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to image or video file.')
    args = parser.parse_args()

    if args.input:
        input_path = args.input.strip()
    else:
        print("\n[Parallel GAN+ViT Deepfake Detector]")
        print("Enter the path to an image or video file.")
        input_path = input("Path: ").strip().strip('"').strip("'")

    if not input_path:
        print("No path provided. Exiting.")
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    run(input_path)


if __name__ == '__main__':
    main()
