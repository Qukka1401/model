import io
import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Классификация цифр и животных",
    layout="wide"
)

# Базовый URL API
API_BASE_URL = "http://127.0.0.1:8000"

DIGIT_API_URL = f"{API_BASE_URL}/predict/digit"
ANIMAL_API_URL = f"{API_BASE_URL}/predict/animal"

# Пути к изображениям с метриками
CONFUSION_PATH = "assets/confusion_matrices.png"
METRICS_PATH = "assets/metrics_comparison.png"


def prepare_image_bytes(image: Image.Image) -> bytes:
    """
    Преобразование изображения в PNG-байты для отправки в API.
    """
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def send_to_api(api_url: str, image: Image.Image) -> dict:
    """
    Отправка изображения в FastAPI.
    """
    image_bytes = prepare_image_bytes(image)

    files = {
        "file": ("image.png", image_bytes, "image/png")
    }

    response = requests.post(api_url, files=files, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()


def show_prediction_result(result: dict):
    """
    Отображение результатов классификации.
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    st.success("Классификация выполнена успешно")
    st.subheader("Результат классификации")
    st.write(f"**Предсказанный класс:** {predicted_class}")
    st.write(f"**Уверенность модели:** {confidence:.4f}")

    df = pd.DataFrame({
        "Класс": list(probabilities.keys()),
        "Вероятность": list(probabilities.values())
    }).sort_values(by="Вероятность", ascending=False)

    st.subheader("Вероятности по классам")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Класс"))


def show_metrics_images():
    """
    Отображение матриц ошибок и сравнительных метрик.
    """
    st.subheader("Оценка качества моделей")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(CONFUSION_PATH):
            st.image(CONFUSION_PATH, caption="Матрицы ошибок моделей", use_container_width=True)
        else:
            st.warning("Файл confusion_matrices.png не найден")

    with col2:
        if os.path.exists(METRICS_PATH):
            st.image(METRICS_PATH, caption="Сравнение метрик моделей", use_container_width=True)
        else:
            st.warning("Файл metrics_comparison.png не найден")


def digit_page():
    st.title("Классификация цифр (MNIST)")

    tabs = st.tabs(["Классификация"])

    with tabs[0]:
        input_mode = st.radio(
            "Выберите способ ввода изображения",
            ["Загрузить изображение", "Нарисовать цифру"]
        )

        image = None

        if input_mode == "Загрузить изображение":
            uploaded_file = st.file_uploader(
                "Загрузите изображение цифры",
                type=["png", "jpg", "jpeg"],
                key="digit_uploader"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Загруженное изображение", width=250)

        else:
            st.write("Нарисуйте цифру на холсте")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=14,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="digit_canvas"
            )

            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("RGB")
                image = img
                st.image(image, caption="Нарисованная цифра", width=250)

        if image is not None:
            if st.button("Классифицировать цифру"):
                with st.spinner("Выполняется классификация..."):
                    try:
                        result = send_to_api(DIGIT_API_URL, image)
                        show_prediction_result(result)
                    except Exception as e:
                        st.error("Ошибка при обращении к API")
                        st.code(str(e))



def animal_page():
    st.title("Классификация животных")

    tabs = st.tabs(["Классификация", "Метрики и матрицы ошибок"])

    with tabs[0]:
        uploaded_file = st.file_uploader(
            "Загрузите изображение животного",
            type=["png", "jpg", "jpeg"],
            key="animal_uploader"
        )

        image = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", width=350)

        if image is not None:
            if st.button("Классифицировать животное"):
                with st.spinner("Выполняется классификация..."):
                    try:
                        result = send_to_api(ANIMAL_API_URL, image)
                        show_prediction_result(result)
                    except Exception as e:
                        st.error("Ошибка при обращении к API")
                        st.code(str(e))

    with tabs[1]:
        show_metrics_images()


# Боковое меню с двумя страницами
st.sidebar.title("Меню")
page = st.sidebar.radio(
    "Выберите страницу",
    [
        "Классификация цифр (MNIST)",
        "Классификация животных"
    ]
)

if page == "Классификация цифр (MNIST)":
    digit_page()
else:
    animal_page()