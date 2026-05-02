from model.model_loader import load_model
from app.preprocess import preprocess_image
import numpy as np

model = load_model()

def predict_defect(image_path):
    img = preprocess_image(image_path)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Defective PCB"
    else:
        return "Good PCB"

from app.gradcam import save_gradcam

save_gradcam(
    model,
    img,
    image_path,
    "uploads/gradcam_result.jpg"
)
