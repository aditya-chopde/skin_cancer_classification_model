from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Flask app
app = Flask(__name__)

# Load model once when server starts
MODEL_PATH = "models/my_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_names = {
    0: 'Actinic Keratoses',   # akiec
    1: 'Basal Cell Carcinoma',# bcc
    2: 'Benign Keratosis-like Lesions', # bkl
    3: 'Dermatofibroma',      # df
    4: 'Melanocytic Nevi',    # nv
    5: 'Melanoma',            # mel
    6: 'Vascular Lesions'     # vasc
}

def predictSkinCaner(filePath):
    # Preprocess the image (same as training)
        img = load_img(filePath, target_size=(28, 28))  # Adjust size if different
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        # Save uploaded file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        predicted_class_name = predictSkinCaner(filepath)

        return render_template("index.html",
                               prediction=predicted_class_name,
                               filename=file.filename)

    return render_template("index.html")

# API endpoint (optional, for external calls)
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    predicted_class_name = predictSkinCaner(file)

    return jsonify({
        # "class_index": int(predicted_class_index),
        "class_name": predicted_class_name
    })

if __name__ == "__main__":
    app.run(debug=True)
