import io
import numpy as np
import tensorflow as tf
from PIL import Image
from google.cloud import storage
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]
BUCKET_NAME = "potato-disease-agrocheck"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Download model from GCS to local tmp storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def load_model():
    global model
    if model is None:
        download_blob(BUCKET_NAME, "models/potatoes.keras", "/tmp/potatoes.keras")
        model = tf.keras.models.load_model("/tmp/potatoes.keras")
        print("Model loaded successfully.")


def add_cors_headers(resp):
    """Add CORS headers to a response"""
    resp.headers.add("Access-Control-Allow-Origin", "*")
    resp.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    resp.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return resp


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))

    # Ensure model is loaded
    load_model()

    # Check if file is uploaded
    if "file" not in request.files:
        resp = make_response(jsonify({"error": "No file uploaded"}), 400)
        return add_cors_headers(resp)

    file = request.files["file"]

    try:
        image = Image.open(file.stream).convert("RGB").resize((256, 256))
        image = np.array(image) / 255.0
        img_array = np.expand_dims(image, 0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        resp = make_response(jsonify({"class": predicted_class, "confidence": confidence}), 200)
        return add_cors_headers(resp)
    except Exception as e:
        resp = make_response(jsonify({"error": str(e)}), 500)
        return add_cors_headers(resp)


@app.route("/ping", methods=["GET"])
def ping():
    resp = make_response(jsonify({"message": "Hello, I am alive"}), 200)
    return add_cors_headers(resp)

