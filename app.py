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

# --- CONFIGURATION ---
MODEL_PATH = "skin_model.h5"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/8c6qg0d0b0vv310l61sma/skin_model.h5?rlkey=9shuwdvh6mhhpklzv2h32roam&st=m57r231b&dl=1"

# --- AUTO-DOWNLOAD LOGIC ---
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Dropbox...")
    response = requests.get(DROPBOX_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete!")
    else:
        print("Error: Could not download model. Check your link.")

# Load the model globally so it stays in memory
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- NEW: HOME ROUTE (Fixes 404 Error) ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "DermaPath-AI Backend is running successfully!",
        "endpoint": "/predict"
    }), 200

def predict_image(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Malignant", float(prediction)
    else:
        return "Benign", float(1 - prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    img = Image.open(file)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    result, confidence = predict_image(img)

    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) # Render uses 10000 often, but PORT env covers it
    app.run(host='0.0.0.0', port=port)