#!/usr/bin/env python3.11
#
# TAPE-style frame quality classification for VHS archives.
# Mirrors the frame classification stage from TAPE's real_world_inference.py:
#   CLIP image/text similarity -> Otsu threshold -> good/bad frame split.
#
import argparse
import contextlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common import ARCHIVE_DIR, BASE, METADATA_DIR


DEFAULT_ARCHIVE = "callahan_01_archive"
DEFAULT_CLIP_MODEL = "RN50x4"
TAPE_PROMPTS = [
    "an image that has lost its tracking is bad",
    "an image that is tearing is bad",
    "an image that is very soft is ok though, don't mark it as bad"
    "an image that's interlaced looking is fine too, or has a comb effect is ok",
    "if an image that has a clear date and time in it, that's not warped then it's ok",
    "an image that has lot its color is bad",
    "there's a date and time that appears doubled, then it's a bad image"
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify VHS frames as good/bad using TAPE's CLIP+Otsu method."
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive stem used for default paths (default: {DEFAULT_ARCHIVE}).",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Video path. Default: ../Archive/<archive>.mkv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output folder. Default: metadata/<archive>/tape_badframe",
    )
    parser.add_argument(
        "--scores-tsv",
        default="",
        help="Optional existing frame_scores.tsv to reuse (skips CLIP scoring).",
    )
    parser.add_argument(
        "--existing-badframes",
        default="",
        help="Optional badframes TSV for comparison. Default: metadata/<archive>/badframes.tsv",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to evaluate (default: 0).",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=-1,
        help="Last frame index to evaluate, inclusive (-1 = video end).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Evaluate every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help=f"CLIP model name for clip.load() (default: {DEFAULT_CLIP_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="CLIP image batch size (default: 16).",
    )
    parser.add_argument(
        "--review-png-dir",
        default="",
        help=(
            "Optional reviewed training PNG root with bad/ and good/ folders. "
            "Default: ../Archive/<archive>_tracking_badframe/review_png"
        ),
    )
    parser.add_argument(
        "--prototype-batch-size",
        type=int,
        default=64,
        help="Batch size for encoding review PNG prototypes (default: 64).",
    )
    parser.add_argument(
        "--finetuned-weights",
        default="",
        help=(
            "Optional Step 19 fine-tuned CLIP checkpoint (.pt). "
            "Default: auto-detect latest models/clip_badframe_finetune/<archive>/**/best.pt"
        ),
    )
    parser.add_argument(
        "--device",
        default="",
        help="Torch device override (examples: cpu, cuda). Default: auto.",
    )
    parser.add_argument(
        "--otsu-bins",
        type=int,
        default=256,
        help="Histogram bins for Otsu thresholding (default: 256).",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("otsu", "quantile", "value"),
        default="otsu",
        help="Threshold mode for bad/good split (default: otsu).",
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=-1.0,
        help="When threshold-mode=quantile, label this fraction as bad (0<rate<1).",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=np.nan,
        help="When threshold-mode=value, explicit threshold; score>=threshold => bad.",
    )
    parser.add_argument(
        "--calibrate-bad-rate-from-existing",
        action="store_true",
        help="In quantile mode, infer bad-rate from existing badframes in the evaluated window.",
    )
    parser.add_argument(
        "--export-bad-png-count",
        type=int,
        default=0,
        help="Export up to N evenly spaced predicted bad frames as PNG (default: 0, disabled).",
    )
    parser.add_argument(
        "--png-output-dir",
        default="",
        help="PNG sample output folder. Default: <output-dir>/badframe_samples",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_badframe_ranges(tsv_path):
    ranges = []
    if not tsv_path or not Path(tsv_path).exists():
        return ranges

    for raw_line in Path(tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        ranges.append((max(0, start), max(0, end)))
    return ranges


def expand_ranges_to_set(ranges, min_frame, max_frame):
    out = set()
    if max_frame < min_frame:
        return out
    for start, end in ranges:
        lo = max(min_frame, int(start))
        hi = min(max_frame, int(end))
        if hi < lo:
            continue
        for frame_idx in range(lo, hi + 1):
            out.add(frame_idx)
    return out


def ranges_from_sorted_frames(frame_ids):
    if not frame_ids:
        return []
    ranges = []
    start = prev = int(frame_ids[0])
    for value in frame_ids[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def otsu_threshold(values, bins=256):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("Cannot compute Otsu threshold on empty array.")

    finite_mask = np.isfinite(vals)
    vals = vals[finite_mask]
    if vals.size == 0:
        raise ValueError("Cannot compute Otsu threshold: all values are non-finite.")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin == vmax:
        return vmin

    hist, edges = np.histogram(vals, bins=max(2, int(bins)), range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]

    between = weight1[:-1] * weight2[1:] * np.square(mean1[:-1] - mean2[1:])
    if between.size == 0:
        return float(np.median(vals))
    best_idx = int(np.argmax(between))
    return float(centers[best_idx])


def list_images(folder):
    path = Path(folder)
    if not path.exists():
        return []
    out = []
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            out.append(child)
    return sorted(out, key=lambda p: p.name.lower())


def encode_review_prototypes(
    clip_model,
    clip_preprocess,
    device,
    bad_dir,
    good_dir,
    batch_size=64,
):
    bad_paths = list_images(bad_dir)
    good_paths = list_images(good_dir)
    if not bad_paths:
        raise RuntimeError(f"No bad prototype images found: {bad_dir}")
    if not good_paths:
        raise RuntimeError(f"No good prototype images found: {good_dir}")

    def encode_paths(paths):
        feats = []
        tensors = []
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        for img_path in paths:
            with Image.open(img_path) as im:
                tensors.append(clip_preprocess(im.convert("RGB")))
            if len(tensors) >= int(batch_size):
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad(), amp_ctx:
                    batch_feats = F.normalize(clip_model.encode_image(batch), dim=-1)
                feats.append(batch_feats.detach().cpu())
                tensors.clear()
        if tensors:
            batch = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad(), amp_ctx:
                batch_feats = F.normalize(clip_model.encode_image(batch), dim=-1)
            feats.append(batch_feats.detach().cpu())
        return torch.cat(feats, dim=0)

    bad_feats = encode_paths(bad_paths)
    good_feats = encode_paths(good_paths)
    bad_proto = F.normalize(bad_feats.mean(dim=0), dim=-1).unsqueeze(0).to(device)
    good_proto = F.normalize(good_feats.mean(dim=0), dim=-1).unsqueeze(0).to(device)
    return bad_proto, good_proto, len(bad_paths), len(good_paths)


class ClipBinaryHead(nn.Module):
    def __init__(self, clip_model, feature_dim):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(int(feature_dim), 1)

    def forward(self, image_tensor):
        features = F.normalize(self.clip_model.encode_image(image_tensor), dim=-1)
        logits = self.head(features).squeeze(1)
        return logits


def resolve_latest_finetuned_weights(archive_name):
    runs_root = BASE / "models" / "clip_badframe_finetune" / archive_name
    if not runs_root.exists():
        return None
    candidates = sorted(
        runs_root.glob("**/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def load_finetuned_clip_model(weights_path, device):
    try:
        import clip
    except Exception as exc:
        raise RuntimeError("clip package is required to load fine-tuned checkpoints.") from exc

    state = torch.load(str(weights_path), map_location=device)
    clip_model_name = state.get("clip_model_name")
    if not clip_model_name:
        raise RuntimeError(f"Invalid checkpoint (missing clip_model_name): {weights_path}")

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    feature_dim = int(state.get("feature_dim", 640))
    model = ClipBinaryHead(clip_model, feature_dim).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, clip_preprocess, state


def encode_batch(clip_model, bad_features, tensors, device, good_features=None):
    batch = torch.stack(tensors, dim=0).to(device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
    with torch.no_grad(), amp_ctx:
        img_feats = F.normalize(clip_model.encode_image(batch), dim=-1)
        bad_sim = img_feats @ bad_features.T
        if good_features is None:
            score = bad_sim.squeeze(1)
        else:
            good_sim = img_feats @ good_features.T
            score = (bad_sim - good_sim).squeeze(1)
    return score.detach().cpu().numpy().astype(float).tolist()


def score_video_frames_finetuned(
    video_path,
    start_frame,
    max_frame,
    frame_step,
    batch_size,
    model,
    clip_preprocess,
    device,
):
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
    pbar = tqdm(total=target_count, desc="Scoring frames", unit="frame")

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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            tensors.append(clip_preprocess(pil))
            tensor_indices.append(frame_idx)
            if len(tensors) >= batch_size:
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
                indices.extend(tensor_indices)
                probs.extend(batch_probs)
                tensors.clear()
                tensor_indices.clear()
            pbar.update(1)
        frame_idx += 1

    if tensors:
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
        indices.extend(tensor_indices)
        probs.extend(batch_probs)

    pbar.close()
    cap.release()
    if not indices:
        raise RuntimeError("No frame scores produced.")
    return total_frames, start, end, indices, probs


def score_video_frames(
    video_path,
    start_frame,
    max_frame,
    frame_step,
    batch_size,
    clip_model,
    clip_preprocess,
    bad_features,
    device,
    good_features=None,
):
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
    pbar = tqdm(total=target_count, desc="Scoring frames", unit="frame")

    indices = []
    scores = []
    tensors = []
    tensor_indices = []

    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if ((frame_idx - start) % step) == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            tensors.append(clip_preprocess(pil))
            tensor_indices.append(frame_idx)
            if len(tensors) >= batch_size:
                batch_scores = encode_batch(
                    clip_model,
                    bad_features,
                    tensors,
                    device,
                    good_features=good_features,
                )
                indices.extend(tensor_indices)
                scores.extend(batch_scores)
                tensors.clear()
                tensor_indices.clear()
            pbar.update(1)
        frame_idx += 1

    if tensors:
        batch_scores = encode_batch(
            clip_model,
            bad_features,
            tensors,
            device,
            good_features=good_features,
        )
        indices.extend(tensor_indices)
        scores.extend(batch_scores)

    pbar.close()
    cap.release()

    if not indices:
        raise RuntimeError("No frame scores produced.")
    return total_frames, start, end, indices, scores


def load_scores_tsv(scores_tsv_path):
    rows = []
    for raw_line in Path(scores_tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("frame\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            frame_idx = int(parts[0])
            score = float(parts[1])
        except ValueError:
            continue
        rows.append((frame_idx, score))

    if not rows:
        raise ValueError(f"No frame/score rows found in {scores_tsv_path}")

    rows.sort(key=lambda x: x[0])
    indices = [int(r[0]) for r in rows]
    scores = [float(r[1]) for r in rows]
    return indices, scores


def pick_evenly_spaced_samples(frame_ids, count):
    ordered = [int(x) for x in frame_ids]
    sample_count = max(0, int(count))
    if sample_count <= 0 or not ordered:
        return []
    if sample_count >= len(ordered):
        return ordered

    chosen = []
    seen = set()
    for idx in np.linspace(0, len(ordered) - 1, num=sample_count, dtype=int).tolist():
        frame_id = int(ordered[int(idx)])
        if frame_id in seen:
            continue
        seen.add(frame_id)
        chosen.append(frame_id)

    # Guard against duplicate linspace indices on small ranges.
    if len(chosen) < sample_count:
        for frame_id in ordered:
            if frame_id in seen:
                continue
            seen.add(frame_id)
            chosen.append(frame_id)
            if len(chosen) >= sample_count:
                break
    return chosen


def export_frame_png_samples(video_path, frame_ids, sample_dir, prefix="bad_sample"):
    sample_dir.mkdir(parents=True, exist_ok=True)
    if not frame_ids:
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for PNG export: {video_path}")

    written = []
    failed_frames = []
    try:
        for sample_idx, frame_idx in enumerate(frame_ids, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                failed_frames.append(int(frame_idx))
                continue

            out_path = sample_dir / f"{prefix}_{sample_idx:02d}_frame_{int(frame_idx)}.png"
            if not cv2.imwrite(str(out_path), frame_bgr):
                failed_frames.append(int(frame_idx))
                continue
            written.append(out_path)
    finally:
        cap.release()
    return written, failed_frames


def write_outputs(output_dir, indices, scores, labels, bad_ranges, summary, note="tape_clip"):
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_scores_path = output_dir / "frame_scores.tsv"
    with frame_scores_path.open("w", encoding="utf-8") as f:
        f.write("frame\tscore\tlabel\n")
        for frame_idx, score, label in zip(indices, scores, labels):
            f.write(f"{frame_idx}\t{score:.8f}\t{label}\n")

    badframes_path = output_dir / "badframes.tsv"
    with badframes_path.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for start, end in bad_ranges:
            f.write(f"{start}\t{end}\t{note}\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return frame_scores_path, badframes_path, summary_path


def main(argv=None):
    args = parse_args(argv)

    archive_name = args.archive.strip()
    if not archive_name:
        raise ValueError("--archive cannot be empty.")

    video_path = Path(args.video) if args.video else (ARCHIVE_DIR / f"{archive_name}.mkv")
    output_dir = Path(args.output_dir) if args.output_dir else (METADATA_DIR / archive_name / "tape_badframe")
    png_output_dir = (
        Path(args.png_output_dir) if args.png_output_dir else (output_dir / "badframe_samples")
    )
    existing_badframes = (
        Path(args.existing_badframes)
        if args.existing_badframes
        else (METADATA_DIR / archive_name / "badframes.tsv")
    )
    review_png_dir = (
        Path(args.review_png_dir)
        if args.review_png_dir
        else (ARCHIVE_DIR / f"{archive_name}_tracking_badframe" / "review_png")
    )
    scores_tsv_path = Path(args.scores_tsv) if args.scores_tsv else None

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if scores_tsv_path:
        if not scores_tsv_path.exists():
            raise FileNotFoundError(f"Scores TSV not found: {scores_tsv_path}")
        indices, scores = load_scores_tsv(scores_tsv_path)
        start_frame = min(indices)
        end_frame = max(indices)
        total_frames = None
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        print(f"Loaded {len(indices)} frame scores from: {scores_tsv_path}")
    else:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        finetuned_weights = (
            Path(args.finetuned_weights)
            if args.finetuned_weights
            else resolve_latest_finetuned_weights(archive_name)
        )
        prototype_counts = {"bad": 0, "good": 0}
        finetuned_weights_path = None
        classifier_source = "tape_prompts"
        if finetuned_weights and Path(finetuned_weights).exists():
            finetuned_weights_path = Path(finetuned_weights)
            print(f"Using fine-tuned CLIP checkpoint: {finetuned_weights_path}")
            finetuned_model, clip_preprocess, ft_state = load_finetuned_clip_model(
                finetuned_weights_path,
                device=device,
            )
            classifier_source = "step19_finetuned_clip"
            total_frames, start_frame, end_frame, indices, scores = score_video_frames_finetuned(
                video_path=video_path,
                start_frame=args.start_frame,
                max_frame=args.max_frame,
                frame_step=args.frame_step,
                batch_size=max(1, int(args.batch_size)),
                model=finetuned_model,
                clip_preprocess=clip_preprocess,
                device=device,
            )
        else:
            try:
                import clip
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import `clip`. Install dependencies with:\n"
                    "  .\\.venv\\Scripts\\python.exe -m pip install openai-clip \"setuptools<81\""
                ) from exc

            print(f"Loading CLIP model: {args.clip_model}")
            # `jit=True` is used in the original TAPE code, but it is incompatible with
            # some modern torch/clip combinations on CPU.
            clip_model, clip_preprocess = clip.load(args.clip_model, device=device, jit=False)
            clip_model.eval()

            bad_features = None
            good_features = None
            if (review_png_dir / "bad").exists() and (review_png_dir / "good").exists():
                try:
                    bad_features, good_features, bad_count, good_count = encode_review_prototypes(
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        device=device,
                        bad_dir=review_png_dir / "bad",
                        good_dir=review_png_dir / "good",
                        batch_size=max(1, int(args.prototype_batch_size)),
                    )
                    prototype_counts = {"bad": int(bad_count), "good": int(good_count)}
                    classifier_source = "review_png_prototypes"
                    print(
                        f"Using review PNG prototypes from {review_png_dir} "
                        f"(bad={bad_count}, good={good_count})"
                    )
                except Exception as exc:
                    print(f"WARNING: prototype mode failed ({exc}); falling back to prompt mode.")

            if bad_features is None:
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
                with torch.no_grad(), amp_ctx:
                    tokenized_prompts = clip.tokenize(TAPE_PROMPTS).to(device)
                    text_features = F.normalize(clip_model.encode_text(tokenized_prompts), dim=-1)
                    bad_features = F.normalize(text_features.mean(dim=0), dim=-1).unsqueeze(0)
                good_features = None

            total_frames, start_frame, end_frame, indices, scores = score_video_frames(
                video_path=video_path,
                start_frame=args.start_frame,
                max_frame=args.max_frame,
                frame_step=args.frame_step,
                batch_size=max(1, int(args.batch_size)),
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                bad_features=bad_features,
                device=device,
                good_features=good_features,
            )

    scores_np = np.asarray(scores, dtype=np.float64)
    evaluated_set = set(indices)

    manual_ranges = []
    manual_bad_eval = set()
    if existing_badframes.exists():
        manual_ranges = parse_badframe_ranges(existing_badframes)
        manual_bad = expand_ranges_to_set(manual_ranges, start_frame, end_frame)
        manual_bad_eval = evaluated_set.intersection(manual_bad)

    threshold_mode = args.threshold_mode
    threshold_source = threshold_mode
    target_bad_rate = None
    if threshold_mode == "otsu":
        threshold = otsu_threshold(np.sort(scores_np), bins=args.otsu_bins)
    elif threshold_mode == "quantile":
        target_bad_rate = float(args.bad_rate)
        if args.calibrate_bad_rate_from_existing:
            if not existing_badframes.exists():
                raise ValueError(
                    "--calibrate-bad-rate-from-existing requested, but existing badframes file does not exist."
                )
            if len(manual_bad_eval) <= 0:
                raise ValueError(
                    "--calibrate-bad-rate-from-existing requested, but no manual bad frames were found in the evaluated window."
                )
            target_bad_rate = float(len(manual_bad_eval) / max(1, len(evaluated_set)))
            threshold_source = "quantile_from_existing_badframes"
        if not (0.0 < target_bad_rate < 1.0):
            raise ValueError(
                "For --threshold-mode quantile, use --bad-rate between 0 and 1 "
                "(or pass --calibrate-bad-rate-from-existing)."
            )
        threshold = float(np.quantile(scores_np, 1.0 - target_bad_rate))
    elif threshold_mode == "value":
        threshold = float(args.threshold_value)
        if not np.isfinite(threshold):
            raise ValueError("--threshold-mode value requires a finite --threshold-value.")
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    labels = ["good" if score < threshold else "bad" for score in scores]
    bad_frames = sorted(frame for frame, label in zip(indices, labels) if label == "bad")
    bad_ranges = ranges_from_sorted_frames(bad_frames)
    requested_png_count = max(0, int(args.export_bad_png_count))
    selected_sample_frames = []
    sample_png_paths = []
    failed_sample_frames = []
    if requested_png_count > 0:
        if not video_path.exists():
            print(f"WARNING: skipping PNG export; video file not found: {video_path}")
        else:
            selected_sample_frames = pick_evenly_spaced_samples(bad_frames, requested_png_count)
            try:
                sample_png_paths, failed_sample_frames = export_frame_png_samples(
                    video_path=video_path,
                    frame_ids=selected_sample_frames,
                    sample_dir=png_output_dir,
                )
            except Exception as exc:
                print(f"WARNING: PNG export failed: {exc}")

    summary = {
        "archive": archive_name,
        "video_path": str(video_path),
        "device": str(device),
        "clip_model": args.clip_model,
        "classifier_source": (
            classifier_source if "classifier_source" in locals() else "scores_tsv"
        ),
        "finetuned_weights_path": (
            str(finetuned_weights_path)
            if "finetuned_weights_path" in locals() and finetuned_weights_path is not None
            else None
        ),
        "review_png_dir": (
            str(review_png_dir)
            if "review_png_dir" in locals()
            else None
        ),
        "prototype_counts": (
            prototype_counts if "prototype_counts" in locals() else None
        ),
        "prompts_count": len(TAPE_PROMPTS),
        "total_video_frames": (None if total_frames is None else int(total_frames)),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end": int(end_frame),
        "evaluated_frame_step": int(max(1, args.frame_step)),
        "evaluated_frames": int(len(indices)),
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "score_threshold": float(threshold),
        "target_bad_rate": (None if target_bad_rate is None else float(target_bad_rate)),
        "score_min": float(np.min(scores_np)),
        "score_max": float(np.max(scores_np)),
        "score_mean": float(np.mean(scores_np)),
        "good_frames": int(sum(1 for x in labels if x == "good")),
        "bad_frames": int(sum(1 for x in labels if x == "bad")),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "png_samples": {
            "requested_bad_samples": int(requested_png_count),
            "selected_bad_frames": int(len(selected_sample_frames)),
            "written_pngs": int(len(sample_png_paths)),
            "failed_frames": [int(x) for x in failed_sample_frames],
            "output_dir": str(png_output_dir) if requested_png_count > 0 else None,
        },
    }
    if threshold_mode == "otsu":
        summary["score_threshold_otsu"] = float(threshold)

    if existing_badframes.exists():
        predicted_bad = set(bad_frames)
        tp = len(predicted_bad.intersection(manual_bad_eval))
        fp = len(predicted_bad - manual_bad_eval)
        fn = len(manual_bad_eval - predicted_bad)
        tn = len(evaluated_set) - tp - fp - fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_badframes),
            "manual_bad_ranges": int(len(manual_ranges)),
            "manual_bad_frames_in_window": int(len(manual_bad_eval)),
            "predicted_bad_frames_in_window": int(len(predicted_bad)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    else:
        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_badframes),
            "note": "file not found; comparison skipped",
        }

    frame_scores_path, badframes_path, summary_path = write_outputs(
        output_dir=output_dir,
        indices=indices,
        scores=scores_np.tolist(),
        labels=labels,
        bad_ranges=bad_ranges,
        summary=summary,
        note=f"tape_clip_{threshold_mode}",
    )

    print(f"Frame scores: {frame_scores_path}")
    print(f"Predicted badframe ranges: {badframes_path}")
    print(f"Summary: {summary_path}")
    if requested_png_count > 0:
        print(f"PNG samples: {png_output_dir} ({len(sample_png_paths)} written)")


if __name__ == "__main__":
    main()
