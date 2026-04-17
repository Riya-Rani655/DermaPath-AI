import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = "skin_model.h5"
# REPLACE THIS with your actual Dropbox link (ending in dl=1)
DROPBOX_URL = "https://www.dropbox.com/scl/fi/8c6qg0d0b0vv310l61sma/skin_model.h5?rlkey=9shuwdvh6mhhpklzv2h32roam&st=no6h2a4r&dl=1"

# 1. Automatic Model Downloader
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading...")
    r = requests.get(DROPBOX_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")

# 2. Load Model
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "message": "DermaPath AI is live!"})

def predict_image(img):
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = img.reshape(1,128,128,3)
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "Malignant", float(prediction)
    else:
        return "Benign", float(1 - prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    result, confidence = predict_image(img)
    return jsonify({"result": result, "confidence": confidence})

if __name__ == "__main__":
    # Use Render's dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)