# app.py

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'models/custom_cnn_v1.keras'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT = 48
IMG_WIDTH = 48

# --- 1. Load the Trained Keras Model ---
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- 2. Define the Prediction Function ---
# This function takes an image as input and returns the model's predictions.
def predict_emotion(input_img):
    # The input from Gradio is a PIL Image, so we process it.
    
    # a. Convert to grayscale and resize
    img = input_img.convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
    
    # b. Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # c. Add batch and channel dimensions to match model's input shape (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # d. Make a prediction
    predictions = model.predict(img_array)
    
    # e. Format the output as a dictionary of {emotion: probability}
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# --- 3. Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Emotions"),
    title="Emotion Recognition from Facial Expressions",
    description="Upload a photo of a face, and the model will predict the emotion."
)

# Launch the web application
iface.launch()