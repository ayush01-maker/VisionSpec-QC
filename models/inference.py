import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "saved_model/model.keras"

model = load_model(MODEL_PATH)

CLASS_NAMES = ["Defective", "Non-Defective"]


def predict(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    print("Prediction:", CLASS_NAMES[predicted_class])

    print("Confidence:", confidence)
