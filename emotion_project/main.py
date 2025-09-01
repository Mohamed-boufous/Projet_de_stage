# main.py

# ==================================
# 1. IMPORTATIONS DES LIBRAIRIES
# ==================================
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
# NOUVEAU : Importer CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from PIL import Image

# ==================================
# 2. CONFIGURATION ET CLASSE CUSTOM
# ==================================

# --- Configuration pour le modèle d'image ---
IMAGE_MODEL_PATH = 'models/sota_emotion_model_final_acc_73.04.keras'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Classe FocalLoss pour le chargement du modèle d'image ---
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = self.alpha * tf.pow(1.0 - p_t, self.gamma) * ce
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# ========================================
# 3. CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ========================================

# --- Modèle de Texte ---
print("Chargement du modèle de détection d'émotions textuelles...")
text_model = load_model("models/emotion_model_improved.keras")
label_encoder = joblib.load(open("models/label_encoder.pkl", "rb"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Modèle de texte chargé.")

# --- Modèle d'Image ---
print("Chargement du modèle de détection d'émotions faciales...")
try:
    custom_objects = {'FocalLoss': FocalLoss}
    image_model = load_model(IMAGE_MODEL_PATH, custom_objects=custom_objects)
    print("✅ Modèle d'image chargé.")
    
    # NOUVEAU : "Préchauffage" du modèle pour des prédictions rapides
    print("🚀 Préchauffage du modèle d'image...")
    dummy_image = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    image_model.predict(dummy_image)
    print("✅ Modèle d'image prêt et rapide !")

except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle d'image : {e}")


# ==================================
# 4. INITIALISATION DE FASTAPI
# ==================================
app = FastAPI(
    title="API de Détection d'Émotions",
    description="Une API complète pour prédire les émotions à partir de texte et d'images.",
    version="1.0.0"
)

# NOUVEAU : Configuration de CORS pour autoriser votre application React
origins = [
    "http://localhost",
    "http://localhost:3000", # Port par défaut pour React
    "http://localhost:5173", # Port par défaut pour Vite
    # Ajoutez ici l'URL de votre application si elle est différente
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"], # Autorise tous les en-têtes
)

# Modèle Pydantic pour la validation des données d'entrée du texte
class TextRequest(BaseModel):
    text: str

# ==================================
# 5. DÉFINITION DES ENDPOINTS
# ==================================

# Le reste de votre code reste identique, il est déjà correct.

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de détection d'émotions. Consultez /docs pour les endpoints."}

@app.post("/predict_text_emotion/")
async def predict_text_emotion(request: TextRequest):
    """
    Prédit l'émotion à partir d'une chaîne de caractères et renvoie toutes les probabilités.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide.")

    # Transformation du texte en embedding
    embedding = embed_model.encode([text])
    
    # Prédiction (on récupère le premier et seul élément du batch)
    pred_probabilities = text_model.predict(embedding)[0]
    
    # Obtenir l'émotion principale
    top_emotion_idx = np.argmax(pred_probabilities)
    top_emotion_label = label_encoder.inverse_transform([top_emotion_idx])[0]
    confidence = float(np.max(pred_probabilities))
    
    # NOUVEAU : Créer un dictionnaire de toutes les prédictions
    all_emotion_labels = label_encoder.classes_
    predictions_dict = {label: float(prob) for label, prob in zip(all_emotion_labels, pred_probabilities)}
    
    # La réponse finale qui contient tout ce que le frontend attend
    return {
        "predicted_emotion": top_emotion_label, 
        "confidence": confidence,
        "predictions": predictions_dict # La clé "predictions" est maintenant présente !
    }


@app.post("/predict_image_emotion/")
async def predict_image_emotion(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire le fichier image. Il est peut-être corrompu.")
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = image_model.predict(img_array)
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    predicted_emotion = max(confidences, key=confidences.get)
    return {
        "predicted_emotion": predicted_emotion,
        "confidences": confidences
    }