#!/usr/bin/env python3.11
#
# Run a trained Ultralytics classifier over an archive video and generate
# badframes outputs consumable by step_6_make_videos.py.
#
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import ARCHIVE_DIR, BASE, METADATA_DIR, require_non_empty


DEFAULT_ARCHIVE = "callahan_01_archive"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Infer bad VHS frames with a trained Ultralytics classifier."
    )
    p.add_argument("--archive", default=DEFAULT_ARCHIVE)
    p.add_argument("--video", default="", help="Default: ARCHIVE_DIR/<archive>.mkv")
    p.add_argument(
        "--weights",
        default="",
        help=(
            "Classifier weights (.pt). "
            "Default: latest best.pt under models/badframe_ultralytics_review/<archive>/runs"
        ),
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Default: ARCHIVE_DIR/<archive>_ai_badframe",
    )
    p.add_argument(
        "--metadata-copy-dir",
        default="",
        help="Default: METADATA_DIR/<archive>/ai_badframe",
    )
    p.add_argument(
        "--metadata-badframes-tsv",
        default="",
        help="Default: METADATA_DIR/<archive>/badframes.ai.tsv",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frame", type=int, default=-1)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--device", default="")
    p.add_argument(
        "--threshold-mode",
        choices=("value", "quantile"),
        default="value",
    )
    p.add_argument(
        "--bad-threshold",
        type=float,
        default=0.50,
        help="Used when --threshold-mode value.",
    )
    p.add_argument(
        "--bad-rate",
        type=float,
        default=-1.0,
        help="Used when --threshold-mode quantile.",
    )
    p.add_argument("--note", default="ai_ultralytics_cls")
    return p.parse_args(argv)


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


def resolve_default_weights(archive_name):
    runs_root = BASE / "models" / "badframe_ultralytics_review" / archive_name / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        runs_root.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_bad_class_index(model_names):
    if isinstance(model_names, dict):
        items = list(model_names.items())
    else:
        items = list(enumerate(list(model_names)))
    lowered = {int(idx): str(name).strip().lower() for idx, name in items}
    bad_idx = next((idx for idx, name in lowered.items() if name == "bad"), None)
    if bad_idx is None:
        raise RuntimeError(f"Could not find class name 'bad' in model names: {model_names}")
    return int(bad_idx)


def predict_bad_probs(model, frames_bgr, bad_class_idx, imgsz, device):
    kwargs = {
        "source": frames_bgr,
        "imgsz": int(imgsz),
        "verbose": False,
        "stream": False,
    }
    if str(device or "").strip():
        kwargs["device"] = str(device).strip()
    results = model.predict(**kwargs)
    out = []
    for result in results:
        probs = getattr(result, "probs", None)
        if probs is None or probs.data is None:
            raise RuntimeError("Classifier output missing probabilities.")
        data = probs.data.detach().cpu().numpy()
        if bad_class_idx >= data.shape[0]:
            raise RuntimeError(f"Bad class index {bad_class_idx} out of bounds for {data.shape}.")
        out.append(float(data[bad_class_idx]))
    return out


def score_video(video_path, model, bad_class_idx, batch_size, imgsz, device, start_frame, max_frame, frame_step):
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
    pbar = tqdm(total=target_count, desc="AI scoring frames", unit="frame")

    indices = []
    probs = []
    batch_frames = []
    batch_indices = []
    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if ((frame_idx - start) % step) == 0:
            batch_frames.append(frame_bgr)
            batch_indices.append(int(frame_idx))
            if len(batch_frames) >= int(batch_size):
                batch_probs = predict_bad_probs(model, batch_frames, bad_class_idx, imgsz, device)
                indices.extend(batch_indices)
                probs.extend(batch_probs)
                pbar.update(len(batch_indices))
                batch_frames.clear()
                batch_indices.clear()
        frame_idx += 1

    if batch_frames:
        batch_probs = predict_bad_probs(model, batch_frames, bad_class_idx, imgsz, device)
        indices.extend(batch_indices)
        probs.extend(batch_probs)
        pbar.update(len(batch_indices))

    pbar.close()
    cap.release()
    if not indices:
        raise RuntimeError("No frame probabilities produced.")
    return total_frames, start, end, indices, probs


def mirror_metadata_outputs(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tsv", ".json"}:
            dst = target_dir / path.relative_to(source_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
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
    video_path = Path(args.video) if args.video else (ARCHIVE_DIR / f"{archive}.mkv")
    output_dir = Path(args.output_dir) if args.output_dir else (ARCHIVE_DIR / f"{archive}_ai_badframe")
    metadata_copy_dir = (
        Path(args.metadata_copy_dir) if args.metadata_copy_dir else (METADATA_DIR / archive / "ai_badframe")
    )
    metadata_badframes_tsv = (
        Path(args.metadata_badframes_tsv) if args.metadata_badframes_tsv else (METADATA_DIR / archive / "badframes.ai.tsv")
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    weights_path = Path(args.weights) if args.weights else resolve_default_weights(archive)
    if not weights_path or not Path(weights_path).exists():
        raise FileNotFoundError("No classifier weights found. Provide --weights explicitly.")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for inference.") from exc

    model = YOLO(str(weights_path))
    bad_class_idx = resolve_bad_class_index(model.names)

    total_frames, start_frame, end_frame, indices, probs = score_video(
        video_path=video_path,
        model=model,
        bad_class_idx=bad_class_idx,
        batch_size=max(1, int(args.batch_size)),
        imgsz=max(32, int(args.imgsz)),
        device=str(args.device),
        start_frame=int(args.start_frame),
        max_frame=int(args.max_frame),
        frame_step=int(args.frame_step),
    )

    probs_np = np.asarray(probs, dtype=np.float64)
    if args.threshold_mode == "value":
        threshold = float(args.bad_threshold)
        if not np.isfinite(threshold):
            raise ValueError("--bad-threshold must be finite.")
    else:
        bad_rate = float(args.bad_rate)
        if not (0.0 < bad_rate < 1.0):
            raise ValueError("--bad-rate must be in (0,1) when --threshold-mode quantile.")
        threshold = float(np.quantile(probs_np, 1.0 - bad_rate))

    labels = ["bad" if p >= threshold else "good" for p in probs]
    bad_frames = sorted(int(fi) for fi, lb in zip(indices, labels) if lb == "bad")
    bad_ranges = ranges_from_sorted_frames(bad_frames)

    summary = {
        "archive": archive,
        "video_path": str(video_path),
        "weights_path": str(weights_path),
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
