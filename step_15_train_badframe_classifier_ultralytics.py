#!/usr/bin/env python3.11
#
# Train an Ultralytics image classifier to separate VHS good/bad frames.
# badframes.tsv defines bad frame ranges; all other frames are treated as good.
#
import argparse
import random
import re
import shutil
from pathlib import Path

from common import ARCHIVE_DIR, BASE, METADATA_DIR


DEFAULT_ARCHIVE = "callahan_01_archive"
DEFAULT_MODEL = "yolo11n-cls.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics classifier from badframes.tsv labels."
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive stem, used for defaults (default: {DEFAULT_ARCHIVE}).",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="Path to source proxy video. Default: ../Archive/<archive>_proxy.mp4",
    )
    parser.add_argument(
        "--badframes",
        default="",
        help="Path to badframes TSV. Default: metadata/<archive>/badframes.tsv",
    )
    parser.add_argument(
        "--output-root",
        default=str(BASE / "models" / "badframe_ultralytics"),
        help="Output root for prepared dataset and training runs.",
    )
    parser.add_argument(
        "--good-bad-ratio",
        type=float,
        default=1.0,
        help="How many good frames to sample per bad frame (<=0 keeps all good frames).",
    )
    parser.add_argument(
        "--max-bad",
        type=int,
        default=0,
        help="Maximum bad frames to use (0 = all).",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=-1,
        help="Only consider frames [0..max_frame] (default: -1 = full video).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training split fraction per class (default: 0.70).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Validation split fraction per class (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling/splitting.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for extracted frames (1-100).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset only, do not start Ultralytics training.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing dataset folder before preparing new images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ultralytics classification model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--device",
        default="",
        help="Ultralytics device string (examples: cpu, 0). Default lets Ultralytics choose.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Training run name. Default: <archive>_badframe_cls",
    )
    return parser.parse_args(argv)


def parse_badframe_ranges(tsv_path):
    ranges = []
    for raw_line in Path(tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cols = re.split(r"\s+", line)
        if len(cols) < 2:
            continue
        try:
            start = int(cols[0])
            end = int(cols[1])
        except ValueError:
            # Header rows or malformed lines are skipped.
            continue
        if end < start:
            start, end = end, start
        ranges.append((max(0, start), max(0, end)))
    if not ranges:
        raise ValueError(f"No bad frame ranges parsed from {tsv_path}")
    return ranges


def expand_ranges(ranges, max_frame):
    bad = set()
    for start, end in ranges:
        if start > max_frame:
            continue
        hi = min(max_frame, end)
        for frame_idx in range(start, hi + 1):
            bad.add(frame_idx)
    return bad


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


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame\tlabel\tsplit\timage_path"]
    for frame_idx, label, split, rel_path in rows:
        lines.append(f"{frame_idx}\t{label}\t{split}\t{rel_path}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(
    archive_name,
    proxy_path,
    badframes_path,
    output_root,
    seed,
    good_bad_ratio,
    max_bad,
    max_frame,
    train_frac,
    val_frac,
    jpeg_quality,
    rebuild,
):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) is required to extract frame images.") from exc

    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {proxy_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read frame count for: {proxy_path}")

    max_allowed_frame = total_frames - 1
    if int(max_frame) >= 0:
        max_allowed_frame = min(max_allowed_frame, int(max_frame))

    bad_ranges = parse_badframe_ranges(badframes_path)
    bad_ids = sorted(expand_ranges(bad_ranges, max_allowed_frame))
    if not bad_ids:
        cap.release()
        raise RuntimeError("No bad frames remain after max-frame filtering.")

    rng = random.Random(int(seed))
    if int(max_bad) > 0 and len(bad_ids) > int(max_bad):
        bad_ids = sorted(rng.sample(bad_ids, int(max_bad)))

    bad_set = set(bad_ids)
    good_candidates = [idx for idx in range(max_allowed_frame + 1) if idx not in bad_set]
    if not good_candidates:
        cap.release()
        raise RuntimeError("No good frames found (all frames are marked bad).")

    if float(good_bad_ratio) > 0:
        target_good = int(round(len(bad_ids) * float(good_bad_ratio)))
        target_good = max(1, target_good)
        if target_good < len(good_candidates):
            good_ids = sorted(rng.sample(good_candidates, target_good))
        else:
            good_ids = list(good_candidates)
    else:
        good_ids = list(good_candidates)

    train_bad, val_bad, test_bad = split_ids(bad_ids, train_frac, val_frac, seed + 11)
    train_good, val_good, test_good = split_ids(good_ids, train_frac, val_frac, seed + 29)

    dataset_dir = Path(output_root) / archive_name / "dataset_cls"
    if rebuild and dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    for split in ("train", "val", "test"):
        for label in ("bad", "good"):
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)

    selected = {}

    def assign(ids, split, label):
        for frame_idx in ids:
            selected[int(frame_idx)] = (split, label)

    assign(train_bad, "train", "bad")
    assign(val_bad, "val", "bad")
    assign(test_bad, "test", "bad")
    assign(train_good, "train", "good")
    assign(val_good, "val", "good")
    assign(test_good, "test", "good")

    wanted_frames = sorted(selected.keys())
    rows = []
    ptr = 0
    frame_idx = 0
    total_wanted = len(wanted_frames)
    next_target = wanted_frames[ptr] if wanted_frames else None

    while ptr < total_wanted:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx == next_target:
            split, label = selected[frame_idx]
            out_dir = dataset_dir / split / label
            out_name = f"{archive_name}_{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            wrote = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not wrote:
                cap.release()
                raise RuntimeError(f"Failed to write image: {out_path}")
            rows.append(
                (
                    frame_idx,
                    label,
                    split,
                    out_path.relative_to(dataset_dir).as_posix(),
                )
            )
            ptr += 1
            next_target = wanted_frames[ptr] if ptr < total_wanted else None
            if len(rows) % 1000 == 0:
                print(f"Extracted {len(rows)} / {total_wanted} labeled frames...")

        frame_idx += 1
        if frame_idx > max_allowed_frame and ptr < total_wanted:
            # No more frames in the selected window.
            break

    cap.release()
    if len(rows) != total_wanted:
        missing = total_wanted - len(rows)
        raise RuntimeError(
            f"Frame extraction ended early. Missing {missing} labeled frame images."
        )

    manifest_path = Path(output_root) / archive_name / "labels_manifest.tsv"
    write_manifest(manifest_path, rows)

    def count_split(label, split):
        return sum(1 for _, row_label, row_split, _ in rows if row_label == label and row_split == split)

    print(f"Prepared dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print("Class counts:")
    for split in ("train", "val", "test"):
        good_n = count_split("good", split)
        bad_n = count_split("bad", split)
        print(f"  {split}: good={good_n}, bad={bad_n}")

    return dataset_dir


def train_ultralytics(dataset_dir, output_root, archive_name, model_name, epochs, imgsz, batch, workers, seed, device, run_name):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for training. Install from requirements.txt.") from exc

    run_project = Path(output_root) / archive_name / "runs"
    run_project.mkdir(parents=True, exist_ok=True)

    final_run_name = run_name.strip() or f"{archive_name}_badframe_cls"
    model = YOLO(model_name)

    train_kwargs = {
        "data": str(dataset_dir),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "project": str(run_project),
        "name": final_run_name,
        "seed": int(seed),
    }
    if device.strip():
        train_kwargs["device"] = device.strip()

    print("Starting Ultralytics training...")
    model.train(**train_kwargs)
    print(f"Training finished. Artifacts: {run_project / final_run_name}")


def main(argv=None):
    args = parse_args(argv)

    proxy_path = Path(args.proxy) if args.proxy else ARCHIVE_DIR / f"{args.archive}_proxy.mp4"
    badframes_path = Path(args.badframes) if args.badframes else METADATA_DIR / args.archive / "badframes.tsv"
    output_root = Path(args.output_root)

    if not proxy_path.exists():
        raise FileNotFoundError(f"Missing proxy video: {proxy_path}")
    if not badframes_path.exists():
        raise FileNotFoundError(f"Missing badframes file: {badframes_path}")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be between 0 and 1.")
    if not (0.0 <= float(args.val_frac) < 1.0):
        raise ValueError("--val-frac must be between 0 and 1.")
    if float(args.train_frac) + float(args.val_frac) >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.0.")
    if not (1 <= int(args.jpeg_quality) <= 100):
        raise ValueError("--jpeg-quality must be in [1..100].")

    dataset_dir = prepare_dataset(
        archive_name=args.archive,
        proxy_path=proxy_path,
        badframes_path=badframes_path,
        output_root=output_root,
        seed=int(args.seed),
        good_bad_ratio=float(args.good_bad_ratio),
        max_bad=int(args.max_bad),
        max_frame=int(args.max_frame),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        jpeg_quality=int(args.jpeg_quality),
        rebuild=bool(args.rebuild),
    )

    if args.prepare_only:
        print("Dataset preparation complete (--prepare-only enabled).")
        return

    train_ultralytics(
        dataset_dir=dataset_dir,
        output_root=output_root,
        archive_name=args.archive,
        model_name=args.model,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        seed=int(args.seed),
        device=str(args.device),
        run_name=str(args.run_name),
    )


if __name__ == "__main__":
    main()
