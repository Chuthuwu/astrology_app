from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpius", "sagittarius", "capricornus",
    "aquarius", "pisces"
]


def load_triangle_database(
    db_path: str = "triangle_db/triangle_descriptor_database_v2.npz",
) -> dict[str, np.ndarray]:
    path = Path(db_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Triangle database not found: {path}\n"
            "Put triangle_descriptor_database_v2.npz inside the triangle_db folder."
        )

    db_npz = np.load(path, allow_pickle=True)
    triangle_database: dict[str, np.ndarray] = {}

    for class_id in db_npz.files:
        class_name = CLASS_NAMES[int(class_id)]
        triangle_database[class_name] = db_npz[class_id].astype(np.float32)

    return triangle_database


TRIANGLE_DATABASE = load_triangle_database()


def clamp_bbox(
    bbox,
    image_width: int,
    image_height: int,
    pad: int = 20,
) -> tuple[int, int, int, int]:
    """Clamp bbox and add the same 20 px padding used in Colab."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image_width, x2 + pad)
    y2 = min(image_height, y2 + pad)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


def detect_stars_for_triangle(
    image_bgr: np.ndarray,
    max_stars: int = 12,
    brightness_percentile: float = 99.5,
    min_area: int = 3,
    max_area: int = 300,
) -> list[tuple[float, float]]:
    """
    Colab-style star detection.
    This matches the notebook function detect_star_centroids().
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # The database was created with top_n=12, so do not use too many stars.
    top_n = min(int(max_stars), 12)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_value = np.percentile(blur, brightness_percentile)
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    stars = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y = centroids[i]

        if min_area <= area <= max_area:
            xi = int(np.clip(round(x), 0, gray.shape[1] - 1))
            yi = int(np.clip(round(y), 0, gray.shape[0] - 1))
            brightness = float(gray[yi, xi])
            score = float(area * brightness)
            stars.append((float(x), float(y), area, brightness, score))

    stars = sorted(stars, key=lambda s: s[4], reverse=True)
    stars = stars[:top_n]

    return [(s[0], s[1]) for s in stars]


def build_triangle_descriptors(
    points: list[tuple[float, float]],
    max_points: int = 12,
) -> np.ndarray:
    """Colab-style triangle descriptor: two normalized side ratios."""
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32)

    points = points[:min(max_points, 12)]
    descriptors = []

    for p1, p2, p3 in combinations(points, 3):
        pts = np.array([p1, p2, p3], dtype=np.float32)

        d1 = np.linalg.norm(pts[0] - pts[1])
        d2 = np.linalg.norm(pts[1] - pts[2])
        d3 = np.linalg.norm(pts[2] - pts[0])

        sides = sorted([d1, d2, d3])

        if sides[-1] <= 0:
            continue

        r1 = sides[0] / sides[2]
        r2 = sides[1] / sides[2]
        descriptors.append([r1, r2])

    if not descriptors:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(descriptors, dtype=np.float32)


def _mean_best_nearest_distance(
    test_features: np.ndarray,
    class_database: np.ndarray,
    top_k: int = 30,
) -> float:
    """
    Colab-style score: nearest-neighbor distances, sorted, mean of best top_k.
    Lower distance is better.
    """
    if len(test_features) == 0 or len(class_database) == 0:
        return float("inf")

    test_features = np.asarray(test_features, dtype=np.float32)
    class_database = np.asarray(class_database, dtype=np.float32)

    nearest_distances = []

    # Number of test triangles is small because top_n <= 12, so this is fast enough.
    for desc in test_features:
        distances = np.linalg.norm(class_database - desc, axis=1)
        nearest_distances.append(float(np.min(distances)))

    nearest_distances = np.array(nearest_distances, dtype=np.float32)
    best_distances = np.sort(nearest_distances)[: min(top_k, len(nearest_distances))]

    return float(np.mean(best_distances))


def match_triangle_descriptors(
    test_descriptors: np.ndarray,
    top_k: int = 30,
) -> tuple[str | None, float, dict[str, float]]:
    """
    Returns:
        best_label, confidence_like_score, raw_distance_scores

    raw_distance_scores are Colab-style distances, where LOWER is better.
    confidence_like_score is only for display in the app, where HIGHER looks better.
    """
    if test_descriptors is None or len(test_descriptors) == 0:
        return None, 0.0, {}

    raw_scores: dict[str, float] = {}

    for class_name, db_descriptors in TRIANGLE_DATABASE.items():
        raw_scores[class_name] = _mean_best_nearest_distance(
            test_descriptors,
            db_descriptors,
            top_k=top_k,
        )

    best_label = min(raw_scores, key=raw_scores.get)
    best_distance = raw_scores[best_label]

    # Convert lower-is-better distance into a higher-is-better app score.
    # This score is for display only; decision logic below follows the Colab rule.
    confidence_like_score = 1.0 / (1.0 + 20.0 * best_distance)

    return best_label, float(confidence_like_score), raw_scores


def triangle_predict_from_image(
    image_bgr: np.ndarray,
    max_stars: int = 12,
    brightness_percentile: float = 99.5,
    min_area: int = 3,
    max_area: int = 300,
) -> dict:
    points = detect_stars_for_triangle(
        image_bgr=image_bgr,
        max_stars=max_stars,
        brightness_percentile=brightness_percentile,
        min_area=min_area,
        max_area=max_area,
    )

    descriptors = build_triangle_descriptors(points)
    label, score, raw_scores = match_triangle_descriptors(descriptors, top_k=30)

    return {
        "label": label,
        "score": score,
        "raw_scores": raw_scores,
        "points": points,
        "num_points": len(points),
        "num_triangles": len(descriptors),
    }


def triangle_predict_from_bbox(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_stars: int = 12,
    brightness_percentile: float = 99.5,
    min_area: int = 3,
    max_area: int = 300,
) -> dict:
    image_height, image_width = image_bgr.shape[:2]

    x1, y1, x2, y2 = clamp_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
        pad=20,
    )

    crop = image_bgr[y1:y2, x1:x2]

    result = triangle_predict_from_image(
        image_bgr=crop,
        max_stars=max_stars,
        brightness_percentile=brightness_percentile,
        min_area=min_area,
        max_area=max_area,
    )

    result["bbox"] = (x1, y1, x2, y2)
    return result


def decide_yolo_triangle_result(
    yolo_label: str | None,
    yolo_confidence: float,
    triangle_label: str | None,
    triangle_score: float,
    triangle_accept_score: float = 0.0,
    yolo_strong_confidence: float = 0.70,
) -> dict:
    """
    Colab-style hybrid decision:
    - If YOLO confidence is high, use YOLO.
    - If YOLO and Triangle agree, accept YOLO.
    - If they disagree, use YOLO but mark uncertain.
    - Triangle only is used only when YOLO has no detection.
    """
    if yolo_label is not None:
        yolo_label = yolo_label.lower()

    if triangle_label is not None:
        triangle_label = triangle_label.lower()

    if yolo_label is not None:
        if yolo_confidence >= yolo_strong_confidence:
            return {
                "label": yolo_label,
                "confidence": yolo_confidence,
                "method": "YOLO high confidence",
            }

        if triangle_label is not None and triangle_label == yolo_label:
            return {
                "label": yolo_label,
                "confidence": yolo_confidence,
                "method": "YOLO + Triangle agree",
            }

        if triangle_label is not None and triangle_label != yolo_label:
            return {
                "label": yolo_label,
                "confidence": yolo_confidence,
                "method": "YOLO selected; Triangle disagrees",
            }

        return {
            "label": yolo_label,
            "confidence": yolo_confidence,
            "method": "YOLO only",
        }

    if triangle_label is not None:
        return {
            "label": triangle_label,
            "confidence": triangle_score,
            "method": "Triangle only",
        }

    return {
        "label": None,
        "confidence": 0.0,
        "method": "No detection",
    }
