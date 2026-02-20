Ar#!/usr/bin/env python3.11
#
# Train and run an AI bad-frame classifier using manually reviewed PNG sets.
# Expected review layout (created by step_16_classify_badframes_tracking_loss.py):
#   metadata/<archive>/tracking_badframe/review_png/
#     bad/*.png   (delete false positives before training)
#     good/*.png
#
# Outputs a step_6-compatible badframes.tsv.
#
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import ARCHIVE_DIR, BASE, METADATA_DIR, require_non_empty


DEFAULT_ARCHIVE = "callahan_01_archive"
DEFAULT_MODEL = "yolo11n-cls.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train an Ultralytics classifier from reviewed bad/good PNG frames, "
            "then infer full-video bad frames and write badframes.tsv."
        )
    )
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--video",
        default="",
        help="Input video path. Default: ../Archive/<archive>.mkv",
    )
    parser.add_argument(
        "--review-dir",
        default="",
        help=(
            "Reviewed PNG root with bad/ and good/ subfolders. "
            "Default: metadata/<archive>/tracking_badframe/review_png"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(BASE / "models" / "badframe_ultralytics_review"),
        help="Model output root (dataset + runs).",
    )
    parser.add_argument(
        "--inference-output-dir",
        default="",
        help="Default: metadata/<archive>/ai_badframe",
    )
    parser.add_argument(
        "--badframes-out",
        default="",
        help=(
            "Badframes TSV output path. Default: <inference-output-dir>/badframes.tsv "
            "(compatible with step_6 --badframes-tsv)"
        ),
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing prepared dataset before rebuilding.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--run-name",
        default="",
        help="Default: <archive>_badframe_review_cls",
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Optional .pt weights path for inference. "
            "If provided with --skip-train, training is skipped."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use --weights for inference.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train model but do not run video inference.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset only, do not train or infer.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frame", type=int, default=-1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--infer-imgsz", type=int, default=224)
    parser.add_argument(
        "--threshold-mode",
        choices=("value", "quantile"),
        default="value",
    )
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=0.50,
        help="Used when --threshold-mode value (bad_prob >= threshold => bad).",
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=-1.0,
        help="Used when --threshold-mode quantile (0 < rate < 1).",
    )
    parser.add_argument(
        "--note",
        default="ai_review_classifier",
        help="Note column value for badframes.tsv rows.",
    )
    return parser.parse_args(argv)


def parse_frame_idx_from_name(path):
    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def list_image_files(folder):
    if not folder.exists():
        return []
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        out.extend(folder.glob(ext))
    return sorted(out)


def split_ids(frame_ids, train_frac, val_frac, seed):
    ids = list(frame_ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return [], [], []
    if n == 1:
        return ids, [], []

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

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def _build_labeled_rows(image_paths, label):
    rows = []
    fallback_counter = 0
    for image_path in image_paths:
        frame_idx = parse_frame_idx_from_name(image_path)
        if frame_idx is None:
            fallback_counter += 1
            frame_idx = -(fallback_counter)
        rows.append(
            {
                "frame": int(frame_idx),
                "label": str(label),
                "source_path": image_path.resolve(),
            }
        )
    return rows


def _write_review_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("frame\tlabel\tsplit\tsource_path\tdataset_path\n")
        for row in rows:
            f.write(
                f"{row['frame']}\t{row['label']}\t{row['split']}\t"
                f"{row['source_path']}\t{row['dataset_path']}\n"
            )


def prepare_dataset_from_review_png(
    archive_name,
    review_dir,
    output_root,
    train_frac,
    val_frac,
    seed,
    rebuild=False,
):
    bad_images = list_image_files(review_dir / "bad")
    good_images = list_image_files(review_dir / "good")
    if not bad_images:
        raise RuntimeError(f"No bad PNG files found in: {review_dir / 'bad'}")
    if not good_images:
        raise RuntimeError(f"No good PNG files found in: {review_dir / 'good'}")

    bad_rows = _build_labeled_rows(bad_images, "bad")
    good_rows = _build_labeled_rows(good_images, "good")

    bad_ids = list(range(len(bad_rows)))
    good_ids = list(range(len(good_rows)))
    train_bad_ids, val_bad_ids, test_bad_ids = split_ids(bad_ids, train_frac, val_frac, seed + 101)
    train_good_ids, val_good_ids, test_good_ids = split_ids(good_ids, train_frac, val_frac, seed + 307)

    dataset_dir = Path(output_root) / archive_name / "dataset_cls_review"
    if rebuild and dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    for split in ("train", "val", "test"):
        for label in ("bad", "good"):
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    def _copy_rows(rows, selected_ids, split_name):
        for idx in selected_ids:
            row = rows[int(idx)]
            src = Path(row["source_path"])
            ext = src.suffix.lower() or ".png"
            dst_name = f"{archive_name}_{row['label']}_{int(row['frame']):08d}{ext}"
            dst = dataset_dir / split_name / row["label"] / dst_name
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "frame": int(row["frame"]),
                    "label": str(row["label"]),
                    "split": split_name,
                    "source_path": str(src),
                    "dataset_path": str(dst.relative_to(dataset_dir).as_posix()),
                }
            )

    _copy_rows(bad_rows, train_bad_ids, "train")
    _copy_rows(bad_rows, val_bad_ids, "val")
    _copy_rows(bad_rows, test_bad_ids, "test")
    _copy_rows(good_rows, train_good_ids, "train")
    _copy_rows(good_rows, val_good_ids, "val")
    _copy_rows(good_rows, test_good_ids, "test")

    manifest_path = Path(output_root) / archive_name / "review_labels_manifest.tsv"
    _write_review_manifest(manifest_path, manifest_rows)

    def _count(label, split):
        return sum(1 for r in manifest_rows if r["label"] == label and r["split"] == split)

    print(f"Prepared dataset: {dataset_dir}")
    print(f"Review manifest: {manifest_path}")
    for split in ("train", "val", "test"):
        print(
            f"  {split}: bad={_count('bad', split)} good={_count('good', split)}"
        )

    return dataset_dir, manifest_path, manifest_rows


def train_ultralytics_classifier(
    archive_name,
    dataset_dir,
    output_root,
    model_name,
    epochs,
    imgsz,
    batch,
    workers,
    seed,
    device,
    run_name,
):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for training.") from exc

    run_project = Path(output_root) / archive_name / "runs"
    run_project.mkdir(parents=True, exist_ok=True)
    final_run_name = str(run_name or "").strip() or f"{archive_name}_badframe_review_cls"

    model = YOLO(model_name)
    kwargs = {
        "data": str(dataset_dir),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "project": str(run_project),
        "name": final_run_name,
        "seed": int(seed),
    }
    if str(device or "").strip():
        kwargs["device"] = str(device).strip()

    print("Starting Ultralytics training...")
    train_result = model.train(**kwargs)

    run_dir = Path(getattr(train_result, "save_dir", run_project / final_run_name))
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        fallback = run_dir / "weights" / "last.pt"
        if fallback.exists():
            best_weights = fallback
        else:
            raise FileNotFoundError(f"Unable to find trained weights in: {run_dir / 'weights'}")

    print(f"Training finished. Run dir: {run_dir}")
    print(f"Weights: {best_weights}")
    return best_weights, run_dir


def resolve_bad_class_index(model_names):
    if isinstance(model_names, dict):
        items = list(model_names.items())
    else:
        items = list(enumerate(list(model_names)))

    lowered = {int(idx): str(name).strip().lower() for idx, name in items}
    bad_idx = next((idx for idx, name in lowered.items() if name == "bad"), None)
    if bad_idx is None:
        raise RuntimeError(
            f"Could not find class name 'bad' in model names: {model_names}"
        )
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
            raise RuntimeError("Classifier output is missing class probabilities.")
        data = probs.data.detach().cpu().numpy()
        if bad_class_idx >= data.shape[0]:
            raise RuntimeError(
                f"Bad class index {bad_class_idx} is outside probability vector shape {data.shape}."
            )
        out.append(float(data[bad_class_idx]))
    return out


def score_video_with_model(
    video_path,
    model,
    bad_class_idx,
    batch_size,
    imgsz,
    device,
    start_frame,
    max_frame,
    frame_step,
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
    pbar = tqdm(total=target_count, desc="AI scoring frames", unit="frame")

    indices = []
    bad_probs = []
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
                probs = predict_bad_probs(model, batch_frames, bad_class_idx, imgsz, device)
                indices.extend(batch_indices)
                bad_probs.extend(probs)
                pbar.update(len(batch_indices))
                batch_frames.clear()
                batch_indices.clear()
        frame_idx += 1

    if batch_frames:
        probs = predict_bad_probs(model, batch_frames, bad_class_idx, imgsz, device)
        indices.extend(batch_indices)
        bad_probs.extend(probs)
        pbar.update(len(batch_indices))

    pbar.close()
    cap.release()

    if not indices:
        raise RuntimeError("No frame probabilities produced from inference.")
    return total_frames, start, end, indices, bad_probs


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


def write_inference_outputs(
    output_dir,
    badframes_out,
    indices,
    bad_probs,
    labels,
    bad_ranges,
    summary,
    note,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    badframes_out.parent.mkdir(parents=True, exist_ok=True)

    frame_scores_tsv = output_dir / "frame_scores.tsv"
    with frame_scores_tsv.open("w", encoding="utf-8") as f:
        f.write("frame\tbad_probability\tlabel\n")
        for frame_idx, bad_prob, label in zip(indices, bad_probs, labels):
            f.write(f"{int(frame_idx)}\t{float(bad_prob):.8f}\t{label}\n")

    with badframes_out.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for start, end in bad_ranges:
            f.write(f"{int(start)}\t{int(end)}\t{note}\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return frame_scores_tsv, badframes_out, summary_path


def main(argv=None):
    args = parse_args(argv)
    archive_name = require_non_empty(args.archive, "archive")

    video_path = Path(args.video) if args.video else (ARCHIVE_DIR / f"{archive_name}.mkv")
    review_dir = (
        Path(args.review_dir)
        if args.review_dir
        else (METADATA_DIR / archive_name / "tracking_badframe" / "review_png")
    )
    output_root = Path(args.output_root)
    inference_output_dir = (
        Path(args.inference_output_dir)
        if args.inference_output_dir
        else (METADATA_DIR / archive_name / "ai_badframe")
    )
    badframes_out = (
        Path(args.badframes_out)
        if args.badframes_out
        else (inference_output_dir / "badframes.tsv")
    )

    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be between 0 and 1.")
    if not (0.0 <= float(args.val_frac) < 1.0):
        raise ValueError("--val-frac must be between 0 and 1.")
    if float(args.train_frac) + float(args.val_frac) >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.0.")

    if bool(args.prepare_only) and bool(args.train_only):
        raise ValueError("--prepare-only and --train-only cannot be used together.")
    if bool(args.skip_train) and not args.weights:
        raise ValueError("--skip-train requires --weights.")

    dataset_dir, review_manifest_path, review_rows = prepare_dataset_from_review_png(
        archive_name=archive_name,
        review_dir=review_dir,
        output_root=output_root,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        rebuild=bool(args.rebuild),
    )

    if args.prepare_only:
        print("Dataset preparation complete (--prepare-only enabled).")
        return

    trained_weights = None
    run_dir = None
    if not args.skip_train:
        trained_weights, run_dir = train_ultralytics_classifier(
            archive_name=archive_name,
            dataset_dir=dataset_dir,
            output_root=output_root,
            model_name=str(args.model),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            workers=int(args.workers),
            seed=int(args.seed),
            device=str(args.device),
            run_name=str(args.run_name),
        )
    if args.train_only:
        print("Training complete (--train-only enabled).")
        if trained_weights is not None:
            print(f"Trained weights: {trained_weights}")
        return

    if trained_weights is None:
        trained_weights = Path(args.weights)
    trained_weights = Path(trained_weights)
    if not trained_weights.exists():
        raise FileNotFoundError(f"Classifier weights not found: {trained_weights}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for inference.") from exc

    model = YOLO(str(trained_weights))
    bad_class_idx = resolve_bad_class_index(model.names)

    total_frames, start_frame, end_frame, indices, bad_probs = score_video_with_model(
        video_path=video_path,
        model=model,
        bad_class_idx=bad_class_idx,
        batch_size=max(1, int(args.infer_batch_size)),
        imgsz=max(32, int(args.infer_imgsz)),
        device=str(args.device),
        start_frame=int(args.start_frame),
        max_frame=int(args.max_frame),
        frame_step=int(args.frame_step),
    )

    bad_probs_np = np.asarray(bad_probs, dtype=np.float64)
    threshold_mode = str(args.threshold_mode)
    if threshold_mode == "value":
        threshold = float(args.bad_threshold)
        if not np.isfinite(threshold):
            raise ValueError("--bad-threshold must be finite.")
    elif threshold_mode == "quantile":
        bad_rate = float(args.bad_rate)
        if not (0.0 < bad_rate < 1.0):
            raise ValueError("--bad-rate must be in (0,1) when --threshold-mode quantile.")
        threshold = float(np.quantile(bad_probs_np, 1.0 - bad_rate))
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    labels = ["bad" if p >= threshold else "good" for p in bad_probs]
    bad_frames = sorted(
        int(frame_idx)
        for frame_idx, label in zip(indices, labels)
        if label == "bad"
    )
    bad_ranges = ranges_from_sorted_frames(bad_frames)

    summary = {
        "archive": archive_name,
        "video_path": str(video_path),
        "review_dir": str(review_dir),
        "review_manifest_path": str(review_manifest_path),
        "review_images_total": int(len(review_rows)),
        "weights_path": str(trained_weights),
        "training_run_dir": (None if run_dir is None else str(run_dir)),
        "total_video_frames": int(total_frames),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end": int(end_frame),
        "evaluated_frame_step": int(max(1, int(args.frame_step))),
        "evaluated_frames": int(len(indices)),
        "threshold_mode": threshold_mode,
        "bad_threshold": float(threshold),
        "score_min": float(np.min(bad_probs_np)),
        "score_max": float(np.max(bad_probs_np)),
        "score_mean": float(np.mean(bad_probs_np)),
        "good_frames": int(sum(1 for x in labels if x == "good")),
        "bad_frames": int(sum(1 for x in labels if x == "bad")),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "output_badframes_tsv": str(badframes_out),
    }
    if threshold_mode == "quantile":
        summary["bad_rate"] = float(args.bad_rate)

    frame_scores_tsv, written_badframes, summary_path = write_inference_outputs(
        output_dir=inference_output_dir,
        badframes_out=badframes_out,
        indices=indices,
        bad_probs=bad_probs,
        labels=labels,
        bad_ranges=bad_ranges,
        summary=summary,
        note=str(args.note),
    )

    print(f"Frame scores: {frame_scores_tsv}")
    print(f"Badframes TSV: {written_badframes}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
