import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras

# 🔹 Load your trained model
model = keras.models.load_model("models/my_model.h5")

# 🔹 Define the class names (same as training)
class_names = {
    0: 'Actinic Keratoses',      # akiec
    1: 'Basal Cell Carcinoma',   # bcc
    2: 'Benign Keratosis-like Lesions',  # bkl
    3: 'Dermatofibroma',         # df
    4: 'Melanocytic Nevi',       # nv
    5: 'Melanoma',               # mel
    6: 'Vascular Lesions'        # vasc
}

# 🔹 Take image path from user
img_path = input("Enter the path to the image: ").strip()

# 🔹 Load and preprocess the image (28x28, NO normalization)
img = load_img(img_path, target_size=(28, 28))  # trained on 28x28 images
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

# 🔹 Make prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class_index]

print("\n--- Prediction Result ---")
print(f"Image: {img_path}")
print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class name: {predicted_class_name}")
