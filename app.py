import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS  # This is the secret ingredient!
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # This line tells Render: "It's okay to talk to GitHub Pages"

# 1. Configuration - Model Path and Dropbox Link
MODEL_PATH = "skin_model.h5"
# Make sure this link ends with dl=1
DROPBOX_URL = "https://www.dropbox.com/scl/fi/8c6qg0d0b0vv310l61sma/skin_model.h5?rlkey=9shuwdvh6mhhpklzv2h32roam&st=no6h2a4r&dl=1"

# 2. Download Model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Dropbox...")
    response = requests.get(DROPBOX_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download complete!")

# 3. Load the Model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Benign', 'Malignant']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    # Updated to 128x128 to match your model training
    img = img.resize((128, 128))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    score = predictions[0][0]
    
    # Logic depends on your model output (Sigmoid vs Softmax)
    # If using 1 output node (Sigmoid):
    result = class_names[1] if score > 0.5 else class_names[0]
    confidence = float(score if score > 0.5 else 1 - score)
    
    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
