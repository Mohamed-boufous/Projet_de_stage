import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# ================================
# 1. Chargement des modèles
# ================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model_improved.keras")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Charger le modèle d'embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ================================
# 2. Fonction de prédiction
# ================================
def predict_emotion(text: str):
    embedding = embed_model.encode([text])
    prediction = model.predict(embedding)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    return emotion, confidence

# ================================
# 3. Exemple d'utilisation
# ================================
if __name__ == "__main__":
    text = "I am happy"
    emotion, confidence = predict_emotion(text)
    print(f"Texte: {text}")
    print(f"Émotion prédite: {emotion} (confiance: {confidence:.4f})")
