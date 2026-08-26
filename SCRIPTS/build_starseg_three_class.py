#!/usr/bin/env python3
# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Collapse a multi-class StarSeg YOLO dataset into three semantic classes."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


SUNFLOWER_LABELS = {"Pycnopodia_helianthoides"}
NOT_SEA_STAR_LABELS = {
    "Strongylocentrotus_purpuratus",
    "Strongylocentrotus_franciscanus",
    "Strongylocentrotus droebachiensis",
    "Unknown_urchin",
    "Bivalve",
    "Sand_dollar",
    "Dead_urchin",
    "Dead_sandDollar",
    "Dead_bivalve",
}
OUTPUT_NAMES = {
    0: "sunflower_star",
    1: "other_sea_star",
    2: "not_sea_star",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REPO = Path(__file__).resolve().parents[1]


def load_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return {int(class_id): name for class_id, name in names.items()}
    return {class_id: name for class_id, name in enumerate(names)}


def output_class(source_name: str) -> int:
    if source_name in SUNFLOWER_LABELS:
        return 0
    if source_name in NOT_SEA_STAR_LABELS:
        return 2
    return 1


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    if mode == "symlink":
        destination.symlink_to(source)
        return "symlink"
    shutil.copy2(source, destination)
    return "copy"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_manifest_by_label(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            Path(row["label_output_path"]).name: row
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Multi-class YOLO dataset root (data.yaml, images/, labels/, manifest.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "DATA" / "starseg",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
    )
    args = parser.parse_args()

    id_to_name = load_names(args.src / "data.yaml")
    source_names = set(id_to_name.values())
    missing_sunflower = SUNFLOWER_LABELS - source_names
    missing_non_stars = NOT_SEA_STAR_LABELS - source_names
    if missing_sunflower or missing_non_stars:
        raise RuntimeError(
            "Required source labels are missing: "
            f"sunflower={sorted(missing_sunflower)}, "
            f"not_sea_star={sorted(missing_non_stars)}"
        )

    source_manifest = source_manifest_by_label(args.src / "manifest.csv")
    build_root = args.out.with_name(args.out.name + ".building")
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    segment_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    hash_splits: dict[str, set[str]] = defaultdict(set)
    duplicate_label_rows_removed = 0

    for split in ("train", "val"):
        label_root = args.src / "labels" / split
        image_root = args.src / "images" / split
        label_paths = sorted(label_root.glob("*.txt"))
        for label_path in label_paths:
            image_matches = [
                path
                for path in image_root.glob(label_path.stem + ".*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(image_matches) != 1:
                raise RuntimeError(
                    f"Expected one image for {label_path}, found {image_matches}"
                )

            rewritten: list[str] = []
            seen_rows: set[str] = set()
            per_image_counts: Counter[str] = Counter()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                source_id = int(float(parts[0]))
                if source_id not in id_to_name:
                    raise RuntimeError(
                        f"Unknown class id {source_id} in {label_path}"
                    )
                target_id = output_class(id_to_name[source_id])
                target_name = OUTPUT_NAMES[target_id]
                output_row = f"{target_id} " + " ".join(parts[1:])
                if output_row in seen_rows:
                    duplicate_label_rows_removed += 1
                    continue
                seen_rows.add(output_row)
                rewritten.append(output_row)
                per_image_counts[target_name] += 1
                segment_counts[target_name] += 1

            source_image = image_matches[0]
            output_image = build_root / "images" / split / source_image.name
            output_label = build_root / "labels" / split / label_path.name
            mode_used = link_or_copy(source_image, output_image, args.link_mode)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(
                "\n".join(rewritten) + ("\n" if rewritten else ""),
                encoding="utf-8",
            )

            source_row = source_manifest.get(label_path.name)
            if source_row is None:
                raise RuntimeError(f"Missing source manifest row for {label_path.name}")
            sha256 = source_row["sha256"]
            hash_splits[sha256].add(split)
            split_counts[split] += 1
            for class_name in per_image_counts:
                image_counts[class_name] += 1
            rows.append(
                {
                    "split": split,
                    "source_image": str(source_image),
                    "source_label": str(label_path),
                    "output_image": str(args.out / "images" / split / source_image.name),
                    "output_label": str(args.out / "labels" / split / label_path.name),
                    "sha256": sha256,
                    "annotation_count": len(rewritten),
                    "class_counts_json": json.dumps(
                        dict(per_image_counts), sort_keys=True
                    ),
                    "link_mode": mode_used,
                    "project_key": source_row.get("project_key", ""),
                    "task_id": source_row.get("task_id", ""),
                    "image_id": source_row.get("image_id", ""),
                }
            )

    leaking_hashes = {
        sha256: sorted(splits)
        for sha256, splits in hash_splits.items()
        if len(splits) > 1
    }
    if leaking_hashes:
        first = next(iter(leaking_hashes.items()))
        raise RuntimeError(f"Duplicate image crosses splits: {first}")

    (build_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(args.out),
                "train": "images/train",
                "val": "images/val",
                "names": OUTPUT_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_csv(build_root / "manifest.csv", rows)
    summary = {
        "source_dataset": str(args.src),
        "output_root": str(args.out),
        "data_yaml": str(args.out / "data.yaml"),
        "class_count": len(OUTPUT_NAMES),
        "class_names": list(OUTPUT_NAMES.values()),
        "source_class_mapping": {
            name: OUTPUT_NAMES[output_class(name)]
            for name in id_to_name.values()
        },
        "total_images": len(rows),
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "positive_images": sum(1 for row in rows if row["annotation_count"] > 0),
        "negative_unlabeled_images": sum(
            1 for row in rows if row["annotation_count"] == 0
        ),
        "segment_counts": dict(segment_counts),
        "class_image_counts": dict(image_counts),
        "duplicate_split_leaks": 0,
        "duplicate_label_rows_removed": duplicate_label_rows_removed,
        "link_mode_requested": args.link_mode,
        "manifest": str(args.out / "manifest.csv"),
    }
    (build_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.out.exists():
        shutil.rmtree(args.out)
    build_root.replace(args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
