#!/usr/bin/env python3.11
#
# Fine-tune CLIP for VHS bad/good frame classification using reviewed PNGs,
# then optionally run full-video inference and emit badframes.tsv.
#
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import ARCHIVE_DIR, BASE, METADATA_DIR, require_non_empty


DEFAULT_ARCHIVE = "callahan_01_archive"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP on reviewed bad/good frames and run inference."
    )
    p.add_argument("--archive", default=DEFAULT_ARCHIVE)
    p.add_argument("--review-png-dir", default="")
    p.add_argument("--clip-model", default="RN50x4")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument(
        "--freeze-clip",
        action="store_true",
        help="Train only classifier head; CLIP weights stay frozen.",
    )
    p.add_argument(
        "--output-root",
        default=str(BASE / "models" / "clip_badframe_finetune"),
    )
    p.add_argument("--run-name", default="")
    p.add_argument("--weights", default="")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--video", default="")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frame", type=int, default=-1)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--infer-batch-size", type=int, default=32)
    p.add_argument("--threshold-mode", choices=("value", "quantile"), default="value")
    p.add_argument("--bad-threshold", type=float, default=0.50)
    p.add_argument("--bad-rate", type=float, default=-1.0)
    p.add_argument("--output-dir", default="")
    p.add_argument("--metadata-copy-dir", default="")
    p.add_argument("--metadata-badframes-tsv", default="")
    p.add_argument("--note", default="clip_finetuned")
    return p.parse_args(argv)


def resolve_device(device_arg):
    if str(device_arg or "").strip():
        return torch.device(str(device_arg).strip())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder):
    path = Path(folder)
    if not path.exists():
        return []
    out = []
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            out.append(child)
    return sorted(out, key=lambda p: p.name.lower())


def split_ids(items, train_frac, val_frac, seed):
    values = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(values)
    n = len(values)
    if n == 0:
        return [], [], []
    if n == 1:
        return values, [], []
    n_train = int(round(n * float(train_frac)))
    n_val = int(round(n * float(val_frac)))
    if n_train <= 0:
        n_train = 1
    if n_train >= n:
        n_train = n - 1
    if n_val < 0:
        n_val = 0
    if n_train + n_val > n:
        n_val = n - n_train
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


class FrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = list(items)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[int(idx)]
        with Image.open(img_path) as im:
            image = self.preprocess(im.convert("RGB"))
        target = torch.tensor(float(label), dtype=torch.float32)
        return image, target


class ClipBinaryHead(nn.Module):
    def __init__(self, clip_model, feature_dim):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(int(feature_dim), 1)

    def forward(self, image_tensor):
        features = F.normalize(self.clip_model.encode_image(image_tensor), dim=-1)
        logits = self.head(features).squeeze(1)
        return logits


def build_items(review_png_dir):
    bad_paths = list_images(Path(review_png_dir) / "bad")
    good_paths = list_images(Path(review_png_dir) / "good")
    if not bad_paths:
        raise RuntimeError(f"No bad images found in: {Path(review_png_dir) / 'bad'}")
    if not good_paths:
        raise RuntimeError(f"No good images found in: {Path(review_png_dir) / 'good'}")
    bad_items = [(p, 1) for p in bad_paths]
    good_items = [(p, 0) for p in good_paths]
    return bad_items + good_items, len(bad_paths), len(good_paths)


def get_feature_dim(clip_model, preprocess, device):
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
    # Use preprocess output shape if model expects a different image size.
    try:
        with Image.new("RGB", (224, 224)) as im:
            sample = preprocess(im).unsqueeze(0).to(device)
            dummy = sample
    except Exception:
        pass
    with torch.no_grad():
        dim = int(clip_model.encode_image(dummy).shape[-1])
    return dim


def train_model(args, device, review_png_dir, checkpoint_dir):
    try:
        import clip
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `clip`. Install with: "
            ".\\.venv\\Scripts\\python.exe -m pip install openai-clip \"setuptools<81\""
        ) from exc

    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.train()

    all_items, bad_count, good_count = build_items(review_png_dir)
    train_items, val_items, test_items = split_ids(
        all_items, args.train_frac, args.val_frac, args.seed
    )
    if not val_items:
        raise RuntimeError("Validation split is empty; adjust train/val fractions.")

    train_loader = DataLoader(
        FrameDataset(train_items, preprocess),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
    )
    val_loader = DataLoader(
        FrameDataset(val_items, preprocess),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
    )

    feature_dim = get_feature_dim(clip_model, preprocess, device)
    model = ClipBinaryHead(clip_model, feature_dim).to(device)

    if bool(args.freeze_clip):
        for param in model.clip_model.parameters():
            param.requires_grad = False

    train_bad = sum(1 for _, y in train_items if int(y) == 1)
    train_good = max(1, sum(1 for _, y in train_items if int(y) == 0))
    pos_weight = torch.tensor([train_good / max(1, train_bad)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for images, targets in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", unit="batch"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_n = int(targets.shape[0])
            train_loss_total += float(loss.item()) * batch_n
            train_count += batch_n
        train_loss = train_loss_total / max(1, train_count)

        model.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Val {epoch}/{args.epochs}", unit="batch"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                batch_n = int(targets.shape[0])
                val_loss_total += float(loss.item()) * batch_n
                val_count += batch_n
        val_loss = val_loss_total / max(1, val_count)

        epoch_row = {"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)}
        history.append(epoch_row)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        state = {
            "archive": str(args.archive),
            "clip_model_name": str(args.clip_model),
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_dim": int(feature_dim),
            "freeze_clip": bool(args.freeze_clip),
            "history": history,
            "review_png_dir": str(review_png_dir),
            "counts": {"bad": int(bad_count), "good": int(good_count)},
            "splits": {
                "train": int(len(train_items)),
                "val": int(len(val_items)),
                "test": int(len(test_items)),
            },
        }
        torch.save(state, last_path)
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            torch.save(state, best_path)

    summary = {
        "clip_model": str(args.clip_model),
        "review_png_dir": str(review_png_dir),
        "counts": {"bad": int(bad_count), "good": int(good_count)},
        "splits": {
            "train": int(len(train_items)),
            "val": int(len(val_items)),
            "test": int(len(test_items)),
        },
        "freeze_clip": bool(args.freeze_clip),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "history": history,
        "best_weights": str(best_path),
        "last_weights": str(last_path),
    }
    (checkpoint_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return best_path, summary


def load_model_for_inference(weights_path, device):
    try:
        import clip
    except Exception as exc:
        raise RuntimeError("clip package is required for inference.") from exc

    state = torch.load(str(weights_path), map_location=device)
    clip_model_name = state["clip_model_name"]
    clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    feature_dim = int(state.get("feature_dim", get_feature_dim(clip_model, preprocess, device)))
    model = ClipBinaryHead(clip_model, feature_dim).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, preprocess, state


def score_video(video_path, model, preprocess, device, start_frame, max_frame, frame_step, batch_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read frame count for: {video_path}")

    start = max(0, int(start_frame))
    end = total_frames - 1 if int(max_frame) < 0 else min(total_frames - 1, int(max_frame))
    step = max(1, int(frame_step))
    if start > end:
        cap.release()
        raise ValueError(f"start-frame ({start}) is after max-frame ({end}).")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    target_count = ((end - start) // step) + 1
    pbar = tqdm(total=target_count, desc="Infer frames", unit="frame")

    indices = []
    probs = []
    tensors = []
    tensor_indices = []
    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if ((frame_idx - start) % step) == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(preprocess(pil))
            tensor_indices.append(int(frame_idx))
            if len(tensors) >= int(batch_size):
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
                indices.extend(tensor_indices)
                probs.extend(batch_probs)
                pbar.update(len(tensor_indices))
                tensors.clear()
                tensor_indices.clear()
        frame_idx += 1

    if tensors:
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
        indices.extend(tensor_indices)
        probs.extend(batch_probs)
        pbar.update(len(tensor_indices))

    pbar.close()
    cap.release()
    if not indices:
        raise RuntimeError("No inference scores produced.")
    return total_frames, start, end, indices, probs


def ranges_from_sorted_frames(frame_ids):
    if not frame_ids:
        return []
    out = []
    start = prev = int(frame_ids[0])
    for v in frame_ids[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        out.append((start, prev))
        start = prev = v
    out.append((start, prev))
    return out


def mirror_metadata_outputs(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tsv", ".json"}:
            dst = target_dir / p.relative_to(source_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            copied.append(dst)
    return copied


def write_outputs(output_dir, indices, probs, labels, bad_ranges, summary, note):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_scores_path = output_dir / "frame_scores.tsv"
    with frame_scores_path.open("w", encoding="utf-8") as f:
        f.write("frame\tbad_probability\tlabel\n")
        for fi, pb, lb in zip(indices, probs, labels):
            f.write(f"{int(fi)}\t{float(pb):.8f}\t{lb}\n")

    badframes_path = output_dir / "badframes.tsv"
    with badframes_path.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for a, b in bad_ranges:
            f.write(f"{int(a)}\t{int(b)}\t{note}\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return frame_scores_path, badframes_path, summary_path


def main(argv=None):
    args = parse_args(argv)
    archive = require_non_empty(args.archive, "archive")
    device = resolve_device(args.device)
    review_png_dir = (
        Path(args.review_png_dir)
        if args.review_png_dir
        else (ARCHIVE_DIR / f"{archive}_tracking_badframe" / "review_png")
    )
    run_name = str(args.run_name or "").strip() or f"{archive}_clip_finetune"
    checkpoint_dir = Path(args.output_root) / archive / run_name

    if bool(args.skip_train) and not str(args.weights or "").strip():
        raise ValueError("--skip-train requires --weights.")

    weights_path = None
    train_summary = None
    if not args.skip_train:
        weights_path, train_summary = train_model(args, device, review_png_dir, checkpoint_dir)
        print(f"Training complete. Best weights: {weights_path}")
    else:
        weights_path = Path(args.weights)

    if args.train_only:
        return

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    video_path = Path(args.video) if args.video else (ARCHIVE_DIR / f"{archive}.mkv")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model, preprocess, loaded_state = load_model_for_inference(weights_path, device)
    total_frames, start_frame, end_frame, indices, probs = score_video(
        video_path=video_path,
        model=model,
        preprocess=preprocess,
        device=device,
        start_frame=int(args.start_frame),
        max_frame=int(args.max_frame),
        frame_step=int(args.frame_step),
        batch_size=max(1, int(args.infer_batch_size)),
    )

    probs_np = np.asarray(probs, dtype=np.float64)
    if args.threshold_mode == "value":
        threshold = float(args.bad_threshold)
        if not np.isfinite(threshold):
            raise ValueError("--bad-threshold must be finite.")
    else:
        bad_rate = float(args.bad_rate)
        if not (0.0 < bad_rate < 1.0):
            raise ValueError("--bad-rate must be in (0,1) for quantile mode.")
        threshold = float(np.quantile(probs_np, 1.0 - bad_rate))

    labels = ["bad" if p >= threshold else "good" for p in probs]
    bad_frames = sorted(int(fi) for fi, lb in zip(indices, labels) if lb == "bad")
    bad_ranges = ranges_from_sorted_frames(bad_frames)

    output_dir = Path(args.output_dir) if args.output_dir else (ARCHIVE_DIR / f"{archive}_clip_finetune_badframe")
    metadata_copy_dir = Path(args.metadata_copy_dir) if args.metadata_copy_dir else (METADATA_DIR / archive / "clip_finetune_badframe")
    metadata_badframes_tsv = (
        Path(args.metadata_badframes_tsv)
        if args.metadata_badframes_tsv
        else (METADATA_DIR / archive / "badframes.clip_finetune.tsv")
    )

    summary = {
        "archive": archive,
        "video_path": str(video_path),
        "device": str(device),
        "weights_path": str(weights_path),
        "clip_model_name": str(loaded_state.get("clip_model_name")),
        "review_png_dir": str(review_png_dir),
        "total_video_frames": int(total_frames),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end": int(end_frame),
        "evaluated_frame_step": int(max(1, int(args.frame_step))),
        "evaluated_frames": int(len(indices)),
        "threshold_mode": str(args.threshold_mode),
        "score_threshold": float(threshold),
        "score_min": float(np.min(probs_np)),
        "score_max": float(np.max(probs_np)),
        "score_mean": float(np.mean(probs_np)),
        "good_frames": int(sum(1 for x in labels if x == "good")),
        "bad_frames": int(sum(1 for x in labels if x == "bad")),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "train_summary": train_summary,
    }
    if args.threshold_mode == "quantile":
        summary["bad_rate"] = float(args.bad_rate)

    frame_scores_path, badframes_path, summary_path = write_outputs(
        output_dir=output_dir,
        indices=indices,
        probs=probs,
        labels=labels,
        bad_ranges=bad_ranges,
        summary=summary,
        note=str(args.note),
    )
    mirrored = mirror_metadata_outputs(output_dir, metadata_copy_dir)
    metadata_badframes_tsv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(badframes_path, metadata_badframes_tsv)

    print(f"Frame scores:           {frame_scores_path}")
    print(f"Predicted bad ranges:   {badframes_path}")
    print(f"Summary:                {summary_path}")
    print(f"Mirrored metadata:      {metadata_copy_dir} ({len(mirrored)} file(s))")
    print(f"Metadata badframes TSV: {metadata_badframes_tsv}")


if __name__ == "__main__":
    main()
