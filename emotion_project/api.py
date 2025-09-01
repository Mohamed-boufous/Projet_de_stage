# api.py: The backend that serves the emotion recognition model.
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = 'models/sota_emotion_model_final_acc_73.04.keras'
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- CRUCIAL: Define FocalLoss class (same as in Gradio version) ---
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
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
        })
        return config

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Emotion Recognition API 🧠",
    description="An API to predict emotions from face images using SOTA model (73.04% accuracy).",
    version="1.0.0"
)

# --- 2. Handle CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load the Model on Startup with FocalLoss ---
print("Loading SOTA model (73.04%) with FocalLoss...")
model = None
try:
    # IMPORTANT: Pass the custom objects dictionary just like in Gradio
    custom_objects = {'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully with FocalLoss!")
except Exception as e:
    print(f"❌ Critical error loading model: {e}")
    raise SystemExit(f"Cannot start server: {e}")

# --- 4. Define Prediction Function (same preprocessing as Gradio) ---
def predict_emotion(image_bytes: bytes):
    """Preprocesses an image exactly like Gradio version and returns emotion probabilities."""
    try:
        input_img = Image.open(io.BytesIO(image_bytes))
        # Same preprocessing as Gradio: convert to RGB and resize
        img = input_img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        # NOTE: Gradio version doesn't normalize by 255.0, so we remove it
        # img_array = img_array / 255.0  # REMOVED - not in Gradio version
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        return confidences
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- 5. Create the API Endpoint ---
@app.post("/predict")
async def handle_prediction(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_bytes = await file.read()
        predictions = predict_emotion(image_bytes)

        if predictions is None:
            raise HTTPException(status_code=500, detail="Internal server error while processing the image.")

        # Format response similar to Gradio output
        sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "success": True,
            "predictions": sorted_predictions,
            "top_emotion": max(predictions, key=predictions.get),
            "confidence": max(predictions.values())
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- 6. Additional endpoint for model info ---
@app.get("/model-info")
def get_model_info():
    return {
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "image_size": [IMG_HEIGHT, IMG_WIDTH],
        "accuracy": "73.04%",
        "architecture": "EfficientNetV2-M with FocalLoss",
        "status": "loaded" if model is not None else "not_loaded"
    }

# --- 7. Root Endpoint (for health check) ---
@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Welcome to the Emotion Recognition API! 🤖",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/model-info"]
    }

# --- To run the server directly ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)