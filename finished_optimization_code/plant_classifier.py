"""
plant_classifier.py — Image-based plant growth stage classification.

Classifies greenhouse camera images into one of three states:
  - GERMINATING: mostly soil visible, tiny/no green shoots
  - GROWING:     visible green plants, active growth
  - HARVEST_READY: large plants, flowers/fruit

Uses OpenCV color analysis (green pixel ratio + plant area).
Interface designed for easy swap to a trained ML model later.

Thresholds can be auto-calibrated from labeled photos using
calibrate_classifier.py (produces classifier_config.json).
"""

import os
import json
import time
import numpy as np
from typing import Optional
from enum import Enum

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class PlantState(Enum):
    GERMINATING = "GERMINATING"
    GROWING = "GROWING"
    HARVEST_READY = "HARVEST_READY"


# Load calibrated thresholds if available
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_config.json")
_CALIBRATED_THRESHOLDS = {}

def _load_calibrated_thresholds():
    """Load thresholds from calibration run if available."""
    global _CALIBRATED_THRESHOLDS
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as f:
                config = json.load(f)
            _CALIBRATED_THRESHOLDS = config.get("thresholds", {})
            print(f"[classifier] Loaded calibrated thresholds from {_CONFIG_FILE}")
        except Exception as e:
            print(f"[classifier] Warning: could not load calibration: {e}")
            _CALIBRATED_THRESHOLDS = {}
    else:
        _CALIBRATED_THRESHOLDS = {}

_load_calibrated_thresholds()


# Cache for classification results
_classification_cache = {
    "state": PlantState.GROWING,
    "green_ratio": 0.0,
    "plant_area_pct": 0.0,
    "picture_timestamp": 0,
    "confidence": 0.0,
}


def classify_from_url(
    picture_url: str,
    picture_timestamp: int = 0,
    timeout_s: float = 15.0,
) -> dict:
    """Download image from URL and classify plant state.

    Returns dict with keys: state, green_ratio, plant_area_pct, confidence
    """
    if picture_timestamp == _classification_cache["picture_timestamp"]:
        return _classification_cache

    if not picture_url or not HAS_REQUESTS or not HAS_PIL:
        return _classification_cache

    try:
        resp = requests.get(picture_url, timeout=timeout_s)
        resp.raise_for_status()
        img_bytes = resp.content
    except Exception as e:
        print(f"[classifier] Failed to download image: {e}")
        return _classification_cache

    result = classify_from_bytes(img_bytes)
    result["picture_timestamp"] = picture_timestamp
    _classification_cache.update(result)
    return result


def classify_from_bytes(img_bytes: bytes) -> dict:
    """Classify plant state from raw image bytes (JPEG)."""
    if not HAS_CV2 or not HAS_PIL:
        return {"state": PlantState.GROWING, "green_ratio": 0.0,
                "plant_area_pct": 0.0, "confidence": 0.0}

    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(pil_img)
    except Exception as e:
        print(f"[classifier] Failed to decode image: {e}")
        return {"state": PlantState.GROWING, "green_ratio": 0.0,
                "plant_area_pct": 0.0, "confidence": 0.0}

    return classify_from_array(img_array)


def classify_from_array(img: np.ndarray) -> dict:
    """Classify plant state from a NumPy RGB image array."""
    if not HAS_CV2:
        return {"state": PlantState.GROWING, "green_ratio": 0.0,
                "plant_area_pct": 0.0, "confidence": 0.0}

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Green mask: H=35-85, S=40-255, V=40-255 (OpenCV H range is 0-179)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    green_pixels = np.count_nonzero(green_mask)
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

    # Plant area: find contours of green regions, filter by size
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sum area of significant contours (filter noise)
    min_contour_area = total_pixels * 0.001  # at least 0.1% of image
    plant_area = sum(
        cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_contour_area
    )
    plant_area_pct = plant_area / total_pixels if total_pixels > 0 else 0.0

    # Classification logic
    state, confidence = _classify(green_ratio, plant_area_pct)

    return {
        "state": state,
        "green_ratio": green_ratio,
        "plant_area_pct": plant_area_pct,
        "confidence": confidence,
    }


def _classify(green_ratio: float, plant_area_pct: float) -> tuple[PlantState, float]:
    """Heuristic classification from green metrics.

    Uses calibrated thresholds if available, otherwise defaults.
    """
    t = _CALIBRATED_THRESHOLDS

    if t:
        # Calibrated thresholds — simple boundary-based classification
        if (green_ratio < t.get("germinating_green_max", 0.03) and
                plant_area_pct < t.get("germinating_area_max", 0.01)):
            return PlantState.GERMINATING, 0.9

        if (green_ratio < t.get("growing_green_max", 0.20) and
                plant_area_pct < t.get("growing_area_max", 0.15)):
            # In growing range
            if green_ratio < t.get("growing_green_min", 0.08):
                # Close to germinating boundary
                return PlantState.GERMINATING, 0.6
            return PlantState.GROWING, 0.7

        if green_ratio >= t.get("harvest_green_min", 0.20):
            return PlantState.HARVEST_READY, 0.7

        return PlantState.GROWING, 0.5

    # Default thresholds (no calibration)
    if green_ratio < 0.03 and plant_area_pct < 0.01:
        return PlantState.GERMINATING, 0.9

    if green_ratio < 0.08 and plant_area_pct < 0.05:
        return PlantState.GERMINATING, 0.6 + 0.3 * (1.0 - green_ratio / 0.08)

    if green_ratio < 0.20 and plant_area_pct < 0.15:
        return PlantState.GROWING, 0.5 + 0.4 * min(green_ratio / 0.20, 1.0)

    if green_ratio >= 0.20 or plant_area_pct >= 0.15:
        return PlantState.HARVEST_READY, 0.5 + 0.4 * min(green_ratio / 0.30, 1.0)

    return PlantState.GROWING, 0.5


def get_cached_state() -> dict:
    """Return the last classification result without re-classifying."""
    return dict(_classification_cache)


def classify_from_file(file_path: str) -> dict:
    """Classify plant state from a local image file.

    Useful for testing with sample photos.
    """
    if not HAS_CV2 or not HAS_PIL:
        return {"state": PlantState.GROWING, "green_ratio": 0.0,
                "plant_area_pct": 0.0, "confidence": 0.0}

    try:
        pil_img = Image.open(file_path).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as e:
        print(f"[classifier] Failed to load image: {e}")
        return {"state": PlantState.GROWING, "green_ratio": 0.0,
                "plant_area_pct": 0.0, "confidence": 0.0}

    return classify_from_array(img_array)


def classify_placeholder() -> dict:
    """Placeholder when image classification is not available.
    Defaults to GROWING state."""
    return {
        "state": PlantState.GROWING,
        "green_ratio": 0.0,
        "plant_area_pct": 0.0,
        "confidence": 0.0,
        "picture_timestamp": 0,
    }
