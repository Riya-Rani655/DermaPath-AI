import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = "skin_model.h5"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/8c6qg0d0b0vv310l61sma/skin_model.h5?rlkey=9shuwdvh6mhhpklzv2h32roam&st=no6h2a4r&dl=1"

# Global variable to hold the model
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            r = requests.get(DROPBOX_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    try:
        # Load model only when needed to save memory at boot
        current_model = get_model()
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128)) 
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = current_model.predict(img_array)
        score = float(predictions[0][0])
        
        result = "Malignant" if score > 0.5 else "Benign"
        confidence = float(score if score > 0.5 else 1 - score)
        
        return jsonify({"result": result, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
