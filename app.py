from __future__ import annotations

from zodiac_data import ZODIAC_DATA
from utils import get_birth_sign
from utils import get_zodiac_result

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="My Star Sign",
   
    page_icon="✨",
    layout="wide",
)

def load_yolo_model(model_path: str) -> YOLO:
    "Load and cache the custom YOLO model."
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Put your trained best.pt file in the models folder, "
            "or enter the correct path in the sidebar."
        )
    return YOLO(str(path))

@st.cache_resource(show_spinner="Loading YOLO model...")
def get_cached_model(model_path: str) -> YOLO:
    return load_yolo_model(model_path)

def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert a Streamlit uploaded image into an OpenCV BGR array."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def clamp_bbox(bbox: Iterable[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Keep an xyxy bounding box inside the image."""
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2

def detect_star_keypoints(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], max_stars: int = 20, brightness_percentile: float = 97.0, min_area: int = 1, max_area: int = 150) -> list[tuple[float, float]]:
    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, image_width=image_width, image_height=image_height)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0: return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    threshold_value = float(np.percentile(enhanced, brightness_percentile))
    _, binary = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    stars: list[dict] = []
    for index in range(1, number_of_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            center_x, center_y = centroids[index]
            region_pixels = enhanced[labels == index]
            if region_pixels.size == 0: continue
            stars.append({"x": float(center_x + x1), "y": float(center_y + y1), "brightness": float(region_pixels.mean()), "area": area})
    stars.sort(key=lambda star: star["brightness"], reverse=True)
    return [(star["x"], star["y"]) for star in stars[:max_stars]]

def draw_star_keypoints(image_bgr: np.ndarray, points: list[tuple[float, float]], start_index: int = 0) -> np.ndarray:
    output = image_bgr.copy()
    for index, (x, y) in enumerate(points, start=start_index):
        center = (int(round(x)), int(round(y)))
        cv2.circle(output, center, 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, str(index), (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return output

def draw_detection_box(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> np.ndarray:
    output = image_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)
    return output

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def main() -> None:
    st.title("Constellation Detection and Star-Keypoint Viewer")
    # ... (giữ nguyên phần Sidebar của bạn) ...
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("YOLO model path", value="models/best.pt")
        confidence = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25, 0.05)
        max_stars = st.slider("Maximum stars per detection", 3, 50, 20, 1)
        brightness_percentile = st.slider("Star brightness percentile", 85.0, 99.9, 97.0, 0.1)
        min_star_area = st.number_input("Minimum star area (pixels)", 1, 100, 1, 1)
        max_star_area = st.number_input("Maximum star area (pixels)", 2, 2000, 150, 5)
        show_keypoints = st.checkbox("Show detected star centers", value=True)
        st.markdown("---")
        birth_month = st.number_input("Birth Month", 1, 12, 1)
        birth_day = st.number_input("Birth Day", 1, 31, 1)

    uploaded_file = st.file_uploader("Upload a night-sky image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    # ... (Xử lý ảnh và model) ...
    image_bgr = uploaded_file_to_bgr(uploaded_file)
    model = get_cached_model(model_path)
    results = model.predict(source=image_bgr, conf=confidence, verbose=False)

    annotated = image_bgr.copy()
    detection_rows = []
    total_keypoints = 0
    zodiac_result = None

    for result in results:
        for box, class_id, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            bbox = clamp_bbox(box, image_bgr.shape[1], image_bgr.shape[0])
            class_name = str(result.names.get(class_id, class_id))
            
            # Zodiac Logic
            detected_sign = class_name.lower()
            SIGN_MAPPING = {"scorpius": "scorpio", "capricornus": "capricorn"}
            detected_sign = SIGN_MAPPING.get(detected_sign, detected_sign)
            zodiac_result = get_zodiac_result(detected_sign, birth_month, birth_day)

            annotated = draw_detection_box(annotated, bbox, f"{class_name} {score:.2f}")
            star_points = detect_star_keypoints(image_bgr, bbox, max_stars, brightness_percentile, int(min_star_area), int(max_star_area))
            
            if show_keypoints:
                annotated = draw_star_keypoints(annotated, star_points)
                total_keypoints += len(star_points)

            detection_rows.append({"Detection": len(detection_rows)+1, "Constellation": class_name, "Confidence": round(float(score), 4), "Detected star centers": len(star_points)})

    # Hiển thị kết quả
    st.image(bgr_to_rgb(annotated))
    
    # ĐÃ SỬA LỖI THỤT LỀ Ở ĐÂY:
    if zodiac_result is not None:
        st.subheader("Personal Zodiac Report")
        st.write(f"Detected Constellation: {zodiac_result['detected_sign']}")
        st.write(f"Birth Zodiac Sign: {zodiac_result['birth_sign']}")

        if zodiac_result["is_birth_match"]:
            st.success(zodiac_result["birth_match_msg"])
        else:
            st.info(zodiac_result["birth_mismatch_msg"])

        st.markdown("---")
        st.subheader("Traits"); st.write(", ".join(zodiac_result["traits"]))
        st.subheader("Strengths"); st.write(", ".join(zodiac_result["strengths"]))
        st.subheader("Weaknesses"); st.write(", ".join(zodiac_result["weaknesses"]))
        st.subheader("Compatible Signs"); st.write(", ".join(zodiac_result["compatible_with"]))
        st.markdown("---")
        st.subheader("Daily Guidance")
        st.write(f"Lucky Color: {zodiac_result['lucky_color']}")
        st.write(f"Daily Horoscope: {zodiac_result['daily_message']}")
        st.write(f"Recommended Food: {zodiac_result['recommended_food']}")
        st.write(f"Suggested Activity: {zodiac_result['activity']}")

        st.markdown("---")

        st.subheader("Astronomical Background")
        st.write(zodiac_result["astronomy_fact"])

        st.subheader("Brightest Star")
        st.write(zodiac_result["brightest_star"])

        st.subheader("Constellation Story")
        st.write(zodiac_result["story"])

        st.subheader("Mythology")
        st.write(zodiac_result["mythology"])

        st.subheader("Fun Fact")
        st.write(zodiac_result["fun_fact"])

        st.subheader("Seen in the Sky Tonight")
        st.info(zodiac_result["seen_in_sky_msg"])
        # ... thêm các thông tin còn lại vào đây ...
    else:
        st.error("No zodiac information found for the detected constellation.")

if __name__ == "__main__":
    main()
