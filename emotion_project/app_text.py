import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Charger les artefacts
model = load_model("models/my_model.h5")
cv = pickle.load(open("models/CountVectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/encoder.pkl", "rb"))

# Fonction de prédiction
def predict_emotion(text):
    if not text.strip():
        return "Veuillez entrer un texte."
    array = cv.transform([text]).toarray()
    pred = model.predict(array)
    label_idx = np.argmax(pred, axis=1)
    emotion = label_encoder.inverse_transform(label_idx)[0]
    return f"Émotion prédite : {emotion}"

# Interface Gradio
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(label="Entrez un texte", placeholder="Exemple: I am happy"),
    outputs=gr.Textbox(label="Résultat"),
    title="Détection des émotions à partir de texte",
    description="Entrez un texte pour prédire l'émotion."
)

# Lancer l'interface
interface.launch()
