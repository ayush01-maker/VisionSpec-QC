from flask import Flask, request, jsonify
from predict import predict_defect
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "VisionSpec-QC API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict_defect(file_path)

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
