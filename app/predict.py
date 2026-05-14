import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "../model/saved_model/model.keras"

model = load_model(MODEL_PATH)

CLASS_NAMES = ["Defective", "Non-Defective"]


def preprocess(img):

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(uploaded_file):

    img = Image.open(uploaded_file).convert("RGB")

    processed = preprocess(img)

    prediction = model.predict(processed)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return CLASS_NAMES[predicted_class], confidence, img
