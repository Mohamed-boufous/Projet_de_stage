from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# Initialiser l'application FastAPI
app = FastAPI(
    title="Emotion Prediction API",
    description="API to predict emotion from text.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Permet à votre frontend React de communiquer avec cette API
origins = [
    "http://localhost:3000", # Port par défaut de React
    "http://localhost:5173", # Port par défaut de Vite (React)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement des modèles ---
# On les charge au démarrage pour ne pas les recharger à chaque requête
try:
    model = load_model("models/emotion_model_improved.keras")
    label_encoder = joblib.load(open("models/label_encoder.pkl", "rb"))
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    # Gérer l'erreur comme vous le souhaitez. Ici, on arrête l'app.
    raise RuntimeError(f"Could not load ML models: {e}")


# --- Modèles de données (Pydantic) ---
# Définit la structure des requêtes et des réponses
class TextRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion: str
    predictions: dict

# --- Endpoint de prédiction ---
@app.post("/predict-text", response_model=EmotionResponse)
def predict_emotion_from_text(request: TextRequest):
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # 1. Transformer le texte en embedding
    embedding = embed_model.encode([text])
    
    # 2. Prédiction avec le modèle Keras
    pred_probabilities = model.predict(embedding)[0]
    
    # 3. Obtenir l'émotion prédite
    label_idx = np.argmax(pred_probabilities)
    emotion = label_encoder.inverse_transform([label_idx])[0]
    
    # 4. (Optionnel) Formater toutes les probabilités pour la réponse
    all_emotions = label_encoder.classes_
    predictions_dict = {
        label_encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(pred_probabilities)
    }

    return EmotionResponse(emotion=emotion, predictions=predictions_dict)

# --- Endpoint racine ---
@app.get("/")
def read_root():
    return {"status": "Emotion Prediction API is running."}