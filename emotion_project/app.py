# app.py - Version finale compatible avec le modèle 69.57%
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION MISE À JOUR ---
MODEL_PATH = 'models/emotion_model_final_acc_69.57.keras' 
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# <-- MODIFIÉ : Taille d'image correspondant au nouveau modèle performant
IMG_HEIGHT = 192
IMG_WIDTH = 192

# --- 1. Charger le Modèle Entraîné ---
print("Chargement du modèle haute performance...")
try:
    # L'option compile=False peut accélérer le chargement pour l'inférence seule
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# --- 2. Définir la Fonction de Prédiction ---
def predict_emotion(input_img: Image.Image):
    """
    Prétraite une image et retourne les probabilités d'émotion.
    """
    # a. Convertir en couleur (RGB) et redimensionner
    img = input_img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    
    # b. Convertir en numpy array (valeurs de 0 à 255, comme pendant l'entraînement)
    img_array = np.array(img)
    
    # c. Ajouter la dimension du lot (batch) pour correspondre à l'entrée du modèle
    # Shape devient : (1, 192, 192, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # d. Faire une prédiction
    predictions = model.predict(img_array)
    
    # e. Formater la sortie en dictionnaire {émotion: probabilité}
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# --- 3. Créer et Lancer l'Interface Gradio ---
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Uploadez une image de visage"),
    outputs=gr.Label(num_top_classes=3, label="Émotions Prédites"),
    title="Analyseur d'Émotions par IA 🤖 (Modèle @69.6%)",
    description="Uploadez une photo de visage et le modèle prédir a l'émotion. Entraîné sur plus de 50,000 images avec EfficientNetV2B2.",
    examples=[["examples/happy_face.jpg"], ["examples/sad_face.jpg"], ["examples/surprise_face.jpg"]],
    allow_flagging="never"
)

# Lancer l'application web
iface.launch(share=True) # share=True pour obtenir un lien public si nécessaire