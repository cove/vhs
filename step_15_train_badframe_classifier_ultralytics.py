#!/usr/bin/env python3.11
#
# Train an Ultralytics image classifier from labeled review PNGs exported by
# step_16/step_17:
#   Archive/<archive>_tracking_badframe/review_png/bad
#   Archive/<archive>_tracking_badframe/review_png/good
#
import argparse
import random
import re
import shutil
from pathlib import Path

from common import ARCHIVE_DIR, BASE


DEFAULT_ARCHIVE = "callahan_01_archive"
DEFAULT_MODEL = "yolo11n-cls.pt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics classifier from review_png good/bad frame images."
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive stem, used for defaults (default: {DEFAULT_ARCHIVE}).",
    )
    parser.add_argument(
        "--review-png-dir",
        default="",
        help="Path to review PNG root with bad/ and good/ folders. "
             "Default: ../Archive/<archive>_tracking_badframe/review_png",
    )
    parser.add_argument(
        "--output-root",
        default=str(BASE / "models" / "badframe_ultralytics_review"),
        help="Output root for prepared dataset and training runs.",
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
        help="Random seed for splitting.",
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
        help="Training run name. Default: <archive>_badframe_review_cls",
    )
    return parser.parse_args(argv)


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

    train_values = values[:n_train]
    val_values = values[n_train:n_train + n_val]
    test_values = values[n_train + n_val:]
    return train_values, val_values, test_values


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame\tlabel\tsplit\tsource_image\timage_path"]
    for frame_idx, label, split, source_path, rel_path in rows:
        lines.append(f"{frame_idx}\t{label}\t{split}\t{source_path}\t{rel_path}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_images(folder):
    return sorted(
        [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def extract_frame_token(path):
    m = re.search(r"(\d+)(?!.*\d)", Path(path).stem)
    return m.group(1) if m else "na"


def prepare_dataset(
    archive_name,
    review_png_dir,
    output_root,
    seed,
    train_frac,
    val_frac,
    rebuild,
):
    review_png_dir = Path(review_png_dir)
    bad_dir = review_png_dir / "bad"
    good_dir = review_png_dir / "good"
    if not bad_dir.exists():
        raise FileNotFoundError(f"Missing bad review folder: {bad_dir}")
    if not good_dir.exists():
        raise FileNotFoundError(f"Missing good review folder: {good_dir}")

    bad_images = list_images(bad_dir)
    good_images = list_images(good_dir)
    if not bad_images:
        raise RuntimeError(f"No bad review images found in: {bad_dir}")
    if not good_images:
        raise RuntimeError(f"No good review images found in: {good_dir}")

    train_bad, val_bad, test_bad = split_ids(bad_images, train_frac, val_frac, seed + 11)
    train_good, val_good, test_good = split_ids(good_images, train_frac, val_frac, seed + 29)

    dataset_dir = Path(output_root) / archive_name / "dataset_cls_review"
    if rebuild and dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    for split in ("train", "val", "test"):
        for label in ("bad", "good"):
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)

    rows = []
    used_names = set()

    def copy_split(images, split, label):
        for idx, src in enumerate(images):
            frame_token = extract_frame_token(src)
            suffix = src.suffix.lower() if src.suffix else ".png"
            base_name = f"{archive_name}_{label}_{frame_token}"
            out_name = f"{base_name}{suffix}"
            if out_name in used_names:
                out_name = f"{base_name}_{idx:04d}{suffix}"
            used_names.add(out_name)

            out_path = dataset_dir / split / label / out_name
            shutil.copy2(src, out_path)
            rows.append(
                (
                    frame_token,
                    label,
                    split,
                    str(src),
                    out_path.relative_to(dataset_dir).as_posix(),
                )
            )

    copy_split(train_bad, "train", "bad")
    copy_split(val_bad, "val", "bad")
    copy_split(test_bad, "test", "bad")
    copy_split(train_good, "train", "good")
    copy_split(val_good, "val", "good")
    copy_split(test_good, "test", "good")

    rows.sort(key=lambda r: (r[2], r[1], r[0], r[4]))
    manifest_path = Path(output_root) / archive_name / "review_labels_manifest.tsv"
    write_manifest(manifest_path, rows)

    def count_split(label, split):
        return sum(1 for _, row_label, row_split, _, _ in rows if row_label == label and row_split == split)

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

    final_run_name = run_name.strip() or f"{archive_name}_badframe_review_cls"
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

    review_png_dir = (
        Path(args.review_png_dir)
        if args.review_png_dir
        else ARCHIVE_DIR / f"{args.archive}_tracking_badframe" / "review_png"
    )
    output_root = Path(args.output_root)

    if not review_png_dir.exists():
        raise FileNotFoundError(f"Missing review PNG root: {review_png_dir}")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be between 0 and 1.")
    if not (0.0 <= float(args.val_frac) < 1.0):
        raise ValueError("--val-frac must be between 0 and 1.")
    if float(args.train_frac) + float(args.val_frac) >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.0.")

    dataset_dir = prepare_dataset(
        archive_name=args.archive,
        review_png_dir=review_png_dir,
        output_root=output_root,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
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
