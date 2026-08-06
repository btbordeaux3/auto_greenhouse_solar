#!/usr/bin/env python3
"""
calibrate_classifier.py — Auto-tune plant classifier thresholds from labeled photos.

Usage:
    1. Create folders with your labeled photos:
       calibration_data/
         germinating/    (photos of seeds/tiny sprouts)
         growing/        (photos of green plants)
         harvest_ready/  (photos of mature/large plants)

    2. Run this script:
       python calibrate_classifier.py

    3. It outputs classifier_config.json with optimal thresholds.
       The plant_classifier.py loads this automatically.

    4. Re-run anytime you collect more photos.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python required.  pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required.  pip install Pillow")
    sys.exit(1)


CALIBRATION_DIR = "calibration_data"
CONFIG_FILE = "classifier_config.json"

CLASSES = ["GERMINATING", "GROWING", "HARVEST_READY"]


def extract_features(img_path: str) -> dict:
    """Extract green pixel ratio and plant area from an image."""
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
    except Exception as e:
        print(f"  Warning: could not load {img_path}: {e}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    green_pixels = np.count_nonzero(green_mask)
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = total_pixels * 0.001
    plant_area = sum(
        cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_contour_area
    )
    plant_area_pct = plant_area / total_pixels if total_pixels > 0 else 0.0

    return {
        "green_ratio": green_ratio,
        "plant_area_pct": plant_area_pct,
        "path": str(img_path),
    }


def compute_class_stats(features: list[dict]) -> dict:
    """Compute statistics for a class from extracted features."""
    if not features:
        return None

    green_ratios = [f["green_ratio"] for f in features]
    area_pcts = [f["plant_area_pct"] for f in features]

    return {
        "green_ratio_mean": float(np.mean(green_ratios)),
        "green_ratio_std": float(np.std(green_ratios)),
        "green_ratio_max": float(np.max(green_ratios)),
        "green_ratio_min": float(np.min(green_ratios)),
        "area_pct_mean": float(np.mean(area_pcts)),
        "area_pct_std": float(np.std(area_pcts)),
        "area_pct_max": float(np.max(area_pcts)),
        "area_pct_min": float(np.min(area_pcts)),
        "n_samples": len(features),
    }


def find_optimal_thresholds(class_stats: dict) -> dict:
    """Find decision boundaries that separate the three classes.

    Uses the 1.5*IQR rule or mean±2*std for robust separation.
    """
    germ = class_stats.get("GERMINATING")
    grow = class_stats.get("GROWING")
    harvest = class_stats.get("HARVEST_READY")

    # Green ratio thresholds
    # GERMINATING: low green, GROWING: medium, HARVEST: high
    if germ and grow and harvest:
        # Boundary between germinating and growing
        # Use midpoint of max(germ) and min(grow), with safety margin
        germ_green_max = germ["green_ratio_max"]
        grow_green_min = grow["green_ratio_min"]
        grow_green_max = grow["green_ratio_max"]
        harvest_green_min = harvest["green_ratio_min"]

        # Germinating/Growing boundary
        if germ_green_max < grow_green_min:
            germinating_growing_boundary = (germ_green_max + grow_green_min) / 2
        else:
            # Overlap — use mean-based approach
            germinating_growing_boundary = (germ["green_ratio_mean"] + grow["green_ratio_mean"]) / 2

        # Growing/Harvest boundary
        if grow_green_max < harvest_green_min:
            growing_harvest_boundary = (grow_green_max + harvest_green_min) / 2
        else:
            growing_harvest_boundary = (grow["green_ratio_mean"] + harvest["green_ratio_mean"]) / 2

        # Area thresholds
        germ_area_max = germ["area_pct_max"]
        grow_area_min = grow["area_pct_min"]
        grow_area_max = grow["area_pct_max"]
        harvest_area_min = harvest["area_pct_min"]

        if germ_area_max < grow_area_min:
            germinating_area_boundary = (germ_area_max + grow_area_min) / 2
        else:
            germinating_area_boundary = (germ["area_pct_mean"] + grow["area_pct_mean"]) / 2

        if grow_area_max < harvest_area_min:
            growing_area_boundary = (grow_area_max + harvest_area_min) / 2
        else:
            growing_area_boundary = (grow["area_pct_mean"] + harvest["area_pct_mean"]) / 2

    elif germ and grow:
        # Only two classes — assume harvest is "lots of green"
        germinating_growing_boundary = (germ["green_ratio_mean"] * 1.5 + grow["green_ratio_mean"] * 0.5) / 2
        growing_harvest_boundary = grow["green_ratio_mean"] * 2.0  # generous
        germinating_area_boundary = (germ["area_pct_mean"] + grow["area_pct_mean"]) / 2
        growing_area_boundary = grow["area_pct_mean"] * 2.0

    else:
        # Fallback to defaults
        print("  Warning: insufficient data for calibration, using defaults")
        return {}

    # Add safety margins (widen the gap by 10%)
    margin_green = 0.10 * max(abs(growing_harvest_boundary - germinating_growing_boundary), 0.02)
    margin_area = 0.10 * max(abs(growing_area_boundary - germinating_area_boundary), 0.01)

    return {
        "germinating_green_max": round(germinating_growing_boundary - margin_green, 4),
        "growing_green_min": round(germinating_growing_boundary + margin_green, 4),
        "growing_green_max": round(growing_harvest_boundary - margin_green, 4),
        "harvest_green_min": round(growing_harvest_boundary + margin_green, 4),
        "germinating_area_max": round(germinating_area_boundary - margin_area, 4),
        "growing_area_min": round(germinating_area_boundary + margin_area, 4),
        "growing_area_max": round(growing_area_boundary - margin_area, 4),
        "harvest_area_min": round(growing_area_boundary + margin_area, 4),
    }


def main():
    print("Plant Classifier Calibration")
    print("=" * 50)

    if not os.path.isdir(CALIBRATION_DIR):
        print(f"ERROR: Create '{CALIBRATION_DIR}/' folder first.")
        print()
        print("Expected structure:")
        print(f"  {CALIBRATION_DIR}/")
        print(f"    germinating/    (photos of seeds/tiny sprouts)")
        print(f"    growing/        (photos of green plants)")
        print(f"    harvest_ready/  (photos of mature plants)")
        print()
        print("Then re-run this script.")
        sys.exit(1)

    class_stats = {}
    all_features = {}

    for cls in CLASSES:
        cls_dir = os.path.join(CALIBRATION_DIR, cls.lower())
        if not os.path.isdir(cls_dir):
            print(f"  Warning: {cls_dir}/ not found, skipping")
            continue

        image_files = [
            os.path.join(cls_dir, f)
            for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not image_files:
            print(f"  Warning: no images in {cls_dir}/")
            continue

        print(f"\n{cls} ({len(image_files)} images):")
        features = []
        for img_path in image_files:
            feat = extract_features(img_path)
            if feat:
                features.append(feat)
                print(f"  {os.path.basename(img_path)}: green={feat['green_ratio']:.3f}  "
                      f"area={feat['plant_area_pct']:.3f}")

        if features:
            stats = compute_class_stats(features)
            class_stats[cls] = stats
            all_features[cls] = features
            print(f"  => green_ratio: {stats['green_ratio_min']:.3f}-{stats['green_ratio_max']:.3f} "
                  f"(mean={stats['green_ratio_mean']:.3f})")
            print(f"  => area_pct:    {stats['area_pct_min']:.3f}-{stats['area_pct_max']:.3f} "
                  f"(mean={stats['area_pct_mean']:.3f})")

    if not class_stats:
        print("\nERROR: No valid images found. Cannot calibrate.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Calibration Results")
    print("=" * 50)

    for cls, stats in class_stats.items():
        print(f"\n{cls} ({stats['n_samples']} samples):")
        for k, v in stats.items():
            if k != "n_samples":
                print(f"  {k}: {v:.4f}")

    thresholds = find_optimal_thresholds(class_stats)

    if thresholds:
        config = {
            "calibration_source": CALIBRATION_DIR,
            "class_stats": {k: v for k, v in class_stats.items()},
            "thresholds": thresholds,
            "total_samples": sum(s["n_samples"] for s in class_stats.values()),
        }

        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved calibration to {CONFIG_FILE}")
        print("\nThresholds for plant_classifier.py:")
        for k, v in thresholds.items():
            print(f"  {k}: {v}")
    else:
        print("\nCould not compute thresholds (insufficient class separation)")

    print("\nDone! The classifier will use these thresholds automatically.")


if __name__ == "__main__":
    main()
