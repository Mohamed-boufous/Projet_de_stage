# app.py - Version finale avec une classe FocalLoss robuste
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'models/sota_emotion_model_final_acc_73.04.keras' 
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- MODIFIÉ : Classe FocalLoss rendue entièrement compatible avec la sauvegarde/chargement ---
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    # On ajoute **kwargs pour accepter les arguments inattendus comme 'reduction'
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        # On passe ces arguments supplémentaires à la classe parente
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = self.alpha * tf.pow(1.0 - p_t, self.gamma) * ce
        return tf.reduce_mean(loss)
    
    # On ajoute get_config pour que Keras sache comment sauvegarder nos paramètres
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config

# --- 1. Charger le Modèle Entraîné ---
print("Chargement du modèle haute performance...")
try:
    # On passe notre classe corrigée à Keras pour qu'il sache la recréer
    custom_objects = {'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects) 
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# --- 2. Définir la Fonction de Prédiction ---
def predict_emotion(input_img: Image.Image):
    img = input_img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    return confidences

# --- 3. Créer et Lancer l'Interface Gradio ---
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Uploadez une image de visage"),
    outputs=gr.Label(num_top_classes=3, label="Émotions Prédites"),
    title="Analyseur d'Émotions par IA 🤖 (Modèle @73%)",
    description="Uploadez une photo de visage et le modèle prédir a l'émotion. Entraîné sur plus de 50,000 images avec une architecture de pointe (EfficientNetV2-M).",
    examples=[["examples/happy_face.jpg"], ["examples/sad_face.jpg"], ["examples.surprise.jpg"]],
    allow_flagging="never"
)

iface.launch(share=True)