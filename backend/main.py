import io
import numpy as np
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Digit and Animal Classification API",
    version="1.0.0",
    description="API для классификации цифр MNIST и изображений животных"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути к моделям
DIGIT_MODEL_PATH = "models/mnist_cnn.keras"
ANIMAL_MODEL_PATH = "models/best_model_final.keras"

# Загрузка моделей
digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
animal_model = tf.keras.models.load_model(ANIMAL_MODEL_PATH)

# Классы
DIGIT_CLASSES = [str(i) for i in range(10)]
ANIMAL_CLASSES = ["cat", "dog", "cheetah"]

# Размеры входов
DIGIT_IMAGE_SIZE = (28, 28)
ANIMAL_IMAGE_SIZE = (224, 224)   # если у вас модель обучалась на другом размере, замените здесь


def preprocess_digit_image(image: Image.Image) -> np.ndarray:
    """
    Предобработка изображения цифры для модели MNIST.
    """
    image = image.convert("L")
    image = image.resize(DIGIT_IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0

    # Для MNIST часто лучше белая цифра на черном фоне
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    img_array = np.expand_dims(img_array, axis=-1)   # (28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)    # (1, 28, 28, 1)

    return img_array


def preprocess_animal_image(image: Image.Image) -> np.ndarray:
    """
    Предобработка изображения животного.
    """
    image = image.convert("RGB")
    image = image.resize(ANIMAL_IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)    # (1, H, W, 3)

    return img_array


def make_prediction(model, image_array: np.ndarray, class_names: list[str]) -> dict:
    """
    Возвращает predicted_class, confidence и probabilities.
    """
    preds = model.predict(image_array, verbose=0)[0]

    # если вдруг модель вернула logits, преобразуем в вероятности
    if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds).numpy()

    predicted_idx = int(np.argmax(preds))
    predicted_class = class_names[predicted_idx]
    confidence = float(preds[predicted_idx])

    probabilities = {
        class_names[i]: float(preds[i]) for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/digit")
async def predict_digit(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    image_array = preprocess_digit_image(image)
    result = make_prediction(digit_model, image_array, DIGIT_CLASSES)

    return {
        "task": "digit",
        "filename": file.filename,
        **result
    }


@app.post("/predict/animal")
async def predict_animal(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    image_array = preprocess_animal_image(image)
    result = make_prediction(animal_model, image_array, ANIMAL_CLASSES)

    return {
        "task": "animal",
        "filename": file.filename,
        **result
    }