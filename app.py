import os
import io
import json
import base64
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

import tensorflow as tf
from tensorflow import keras


st.set_page_config(
    page_title="App",
    layout="wide"
)

DEFAULT_MODEL_PATH = "model.keras"
DEFAULT_LABELS_PATH = "labels.json"

IMG_SIZE = (128, 128)

MAX_PREVIEW_PX = 512


CLASS_TRANSLATIONS = {
    "Blazer": "Піджак",
    "Blouse": "Блузка",
    "Body": "Боді",
    "Dress": "Сукня",
    "Hat": "Головний убір",
    "Hoodie": "Худі",
    "Longsleeve": "Футболка з довгим рукавом",
    "Other": "Інше",
    "Outwear": "Верхній одяг",
    "Pants": "Штани",
    "Polo": "Футболка Поло",
    "Shirt": "Сорочка",
    "Shoes": "Взуття",
    "Shorts": "Шорти",
    "Skip": "Легкий, активний одяг",
    "Skirt": "Спідниця",
    "T-Shirt": "Футболка",
    "Top": "Топ",
    "Undershirt": "Майка",
    "Others": "Інше"
}


def get_display_label(label: str) -> str:
    ukr = CLASS_TRANSLATIONS.get(label)
    return ukr if ukr else label


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep).strip()


@st.cache_resource
def load_model_cached(model_path: str):
    return keras.models.load_model(model_path)


@st.cache_resource
def load_class_names(labels_path: str) -> List[str]:
    lp = labels_path.lower()

    if lp.endswith(".json"):
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data

        if isinstance(data, dict):
            if "labels" in data and isinstance(data["labels"], list) and all(isinstance(x, str) for x in data["labels"]):
                return data["labels"]
            for key in ("class_names", "classes"):
                if key in data and isinstance(data[key], list) and all(isinstance(x, str) for x in data[key]):
                    return data[key]

        raise ValueError("labels.json має містити list[str] або dict з ключем 'labels': list[str].")

    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise ValueError("Файл класів порожній або не прочитався.")

    return classes


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def preprocess_image_for_model(
    img: Image.Image,
    target_size: Tuple[int, int] = (128, 128),
    *,
    center_crop: bool = False,
    auto_contrast: bool = False,
    contrast_boost: float = 1.0,
) -> np.ndarray:
    img = img.convert("RGB")

    if center_crop:
        img = center_crop_to_square(img)

    if auto_contrast:
        img = ImageOps.autocontrast(img)

    if contrast_boost and contrast_boost != 1.0:
        img = ImageEnhance.Contrast(img).enhance(float(contrast_boost))

    img = img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_top1(model, x: np.ndarray, class_names: List[str]):
    proba = model.predict(x, verbose=0)[0]
    i = int(np.argmax(proba))
    if i >= len(class_names):
        return "Unknown", 0.0
    return class_names[i], float(proba[i])


def init_buckets(class_names: List[str]):
    if "buckets" not in st.session_state:
        st.session_state.buckets = {name: [] for name in class_names}
        if "Others" not in st.session_state.buckets:
            st.session_state.buckets["Others"] = []


def add_to_bucket(bucket_name: str, image_bytes: bytes, conf: float, top_label: str):
    if bucket_name not in st.session_state.buckets:
        st.session_state.buckets[bucket_name] = []

    st.session_state.buckets[bucket_name].append({
        "image_bytes": image_bytes,
        "confidence": conf,
        "top_label": top_label
    })


def clear_buckets(class_names: List[str]):
    st.session_state.buckets = {name: [] for name in class_names}
    st.session_state.buckets["Others"] = []


def make_preview_bytes(img: Image.Image, max_px: int = 0, quality: int = 85) -> bytes:
    prev = img.convert("RGB").copy()
    prev.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    prev.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def show_preview_fixed(preview_bytes: bytes, caption: str = "", width_px: int = 0):
    b64 = base64.b64encode(preview_bytes).decode("utf-8")
    html = f"""
    <div style="width:{width_px}px;">
      <img src="data:image/jpeg;base64,{b64}" style="/*! width:{width_px}px; max-width:{width_px}px; height:auto; display:block;" />
      <div style="font-size:12px; color:#666; margin-top:6px;">{caption}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


with st.sidebar:
    st.header("Налаштування")

    model_path_in = st.text_input("Шлях до моделі (.keras)", value=DEFAULT_MODEL_PATH)
    labels_path_in = st.text_input("Шлях до класів (.json)", value=DEFAULT_LABELS_PATH)

    model_path = normalize_path(model_path_in)
    labels_path = normalize_path(labels_path_in)

    st.subheader("Пороги сортування")
    base_threshold = st.slider("Поріг впевненості для сортування", 0.05, 0.95, 0.50, 0.01)

    st.subheader("Препроцесинг")
    center_crop = st.checkbox("Center-crop до квадрата", value=False)
    auto_contrast = st.checkbox("Auto-contrast", value=False)
    contrast_boost = st.slider("Підсилення контрасту", 0.80, 2.00, 1.00, 0.05)

    st.divider()
    if st.button("Очистити всі контейнери"):
        st.session_state["_clear_requested"] = True


try:
    class_names = load_class_names(labels_path)
    model = load_model_cached(model_path)
except Exception as e:
    st.error("Не вдалося завантажити модель або файл класів. Перевір шляхи та наявність файлів.")
    st.exception(e)
    st.stop()

if st.session_state.get("_clear_requested", False):
    clear_buckets(class_names)
    st.session_state["_clear_requested"] = False
    st.success("Контейнери очищено.")

init_buckets(class_names)

tab1, tab2 = st.tabs(["Класифікація", "Сортування"])


with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Завантаження фото")
        uploaded = st.file_uploader("")
        img = None
        img_bytes = None
        preview_bytes = None

    if uploaded is not None:
        filename = uploaded.name.lower()
        mime = uploaded.type

        allowed_ext = (".jpg", ".jpeg", ".png", ".jfif")
        allowed_mime = ("image/jpeg", "image/png")

        if not filename.endswith(allowed_ext):
            st.error("Непідтримуваний формат файлу. Дозволені: JPG, JPEG, PNG.")
    
        elif mime not in allowed_mime:
            st.error(" Файл не є зображенням допустимого формату.")
    
        else:
            try:
                img_bytes = uploaded.read()
                img = Image.open(io.BytesIO(img_bytes))

                preview_bytes = make_preview_bytes(img, MAX_PREVIEW_PX)
                show_preview_fixed(
                    preview_bytes,
                    caption="Вхідне зображення",
                    width_px=MAX_PREVIEW_PX
                )

            except Exception:
                st.error("Неможливо обробити файл як зображення.")

    with col_right:
        st.subheader("Результат")

        if img is None:
            st.info("Завантажте фото зліва, щоб отримати прогноз.")
        else:
            x = preprocess_image_for_model(
                img,
                target_size=IMG_SIZE,
                center_crop=center_crop,
                auto_contrast=auto_contrast,
                contrast_boost=contrast_boost,
            )

            top1_label, top1_conf = predict_top1(model, x, class_names)
            bucket = top1_label if top1_conf >= base_threshold else "Others"

            display_label = get_display_label(top1_label)
            display_bucket = get_display_label(bucket)

            st.write(f"Тип одягу: **{display_label}**")
            st.write(f"Ймовірність: **{top1_conf:.3f}**")

            unique_key = f"{uploaded.name}_{len(img_bytes)}_{bucket}_{top1_label}_{top1_conf:.6f}" if uploaded else None
            last_key = st.session_state.get("_last_auto_added_key")

            if unique_key and last_key != unique_key:
                add_to_bucket(bucket, img_bytes, top1_conf, top1_label)
                st.session_state["_last_auto_added_key"] = unique_key
                st.success(f"Додано у контейнер: {display_bucket}")


with tab2:
    total = sum(len(v) for v in st.session_state.buckets.values())
    others = len(st.session_state.buckets.get("Others", []))
    sorted_count = total - others

    c1, c2, c3 = st.columns(3)
    c1.metric("Всього", total)
    c2.metric("Відсортовано", sorted_count)
    c3.metric("Невизначені (Others)", others)

    st.divider()

    all_buckets = class_names + ["Others"]

    for bucket_name in all_buckets:
        items = st.session_state.buckets.get(bucket_name, [])
        if not items:
            continue

        display_name = get_display_label(bucket_name)
        st.markdown(f"### {display_name} — {len(items)} шт.")

        cols = st.columns(6)
        for i, item in enumerate(items):
            with cols[i % 6]:
                im = Image.open(io.BytesIO(item["image_bytes"]))
                prev_b = make_preview_bytes(im, MAX_PREVIEW_PX)
                show_preview_fixed(prev_b, caption=f"Ймовірність: {item['confidence']:.2f}", width_px=MAX_PREVIEW_PX)
