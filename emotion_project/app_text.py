import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# Charger les artefacts
model = load_model("models/emotion_model_improved.keras")
label_encoder = joblib.load(open("models/label_encoder.pkl", "rb"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ================================
# 3. Fonction de prédiction
# ================================
def predict_emotion(text):
    if not text.strip():
        return "Veuillez entrer un texte."
    
    # Transformer le texte en embedding
    embedding = embed_model.encode([text])
    
    # Prédiction
    pred = model.predict(embedding)
    label_idx = np.argmax(pred)
    emotion = label_encoder.inverse_transform([label_idx])[0]
    
    return f"Émotion prédite : {emotion}"

# ================================
# 4. Interface Gradio
# ================================
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(label="Entrez un texte", placeholder="Exemple: I am happy"),
    outputs=gr.Textbox(label="Résultat"),
    title="Détection des émotions à partir de texte",
    description="Entrez un texte pour prédire l'émotion."
)

# Lancer l'interface
interface.launch()