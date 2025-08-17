# app.py
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION MISE À JOUR ---
# <-- MODIFIÉ : Chemin vers votre nouveau modèle performant
MODEL_PATH = 'models/emotion_model_final_optimized_65.22.keras' 
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# <-- MODIFIÉ : Taille d'image correspondant au nouveau modèle
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- 1. Charger le Modèle Entraîné ---
print("Chargement du modèle optimisé...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    # Quitter si le modèle ne peut pas être chargé
    exit()

# --- 2. Définir la Fonction de Prédiction (Corrigée) ---
def predict_emotion(input_img):
    # a. Convertir en couleur (RGB) et redimensionner
    # <-- MODIFIÉ : .convert('RGB') au lieu de 'L' (grayscale)
    img = input_img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    
    # b. Convertir en numpy array
    # <-- MODIFIÉ : On ne divise PAS par 255.0. Le modèle s'attend à des valeurs de 0 à 255.
    img_array = np.array(img)
    
    # c. Ajouter la dimension du lot (batch) pour correspondre à l'entrée du modèle (1, 128, 128, 3)
    # <-- MODIFIÉ : Pas besoin d'ajouter la dimension du canal, elle est déjà présente (RGB)
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
    title="Reconnaissance des Émotions (Modèle Haute Performance)",
    description="Uploadez une photo de visage, et le modèle prédira l'émotion. Ce modèle a été entraîné sur plus de 50,000 images de 3 datasets différents.",
    examples=[["examples/happy_face.jpg"], ["examples/sad_face.jpg"]] # Optionnel : ajoutez des exemples
)

# Lancer l'application web
iface.launch()