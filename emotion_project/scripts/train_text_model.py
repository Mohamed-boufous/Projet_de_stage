import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import joblib
import sys
from tensorflow.keras.models import save_model

# 1. Chargement du dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "texts", "cleaned_emotion_dataset.csv")
DATA_PATH = os.path.normpath(DATA_PATH)

df = pd.read_csv(DATA_PATH)

# 2. Encodage des labels
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # projet racine
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from preprocessing_text import fit_label_encoder
label_encoder, df['N_label'] = fit_label_encoder(df['Emotion'])
y = df['N_label']

# 3. Embeddings avec SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embed_model.encode(df["Clean_Text"].tolist(), show_progress_bar=True)

# 4. Split Train / Validation / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.25, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# 5. Modèle Keras 
def create_improved_model(input_dim, num_classes):
    model = Sequential([
        # Couche d'entrée avec régularisation L2
        Dense(256, input_shape=(input_dim,), activation='relu', 
              kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Couche cachée 1
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Couche cachée 2
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Couche de sortie
        Dense(num_classes, activation='softmax')
    ])
    return model

# Créer le modèle
num_classes = len(np.unique(y_train))
print(f"Création du modèle avec {num_classes} classes")
model = create_improved_model(X_train.shape[1], num_classes)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Callbacks + Class Weights
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-4, verbose=1)
]

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 7. Entraînement
print("Entraînement du modèle...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# 8. Évaluation
print("\n=== RÉSULTATS ===")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Train - Acc: {train_acc:.4f}")
print(f"Val   - Acc: {val_acc:.4f}")
print(f"Test  - Acc: {test_acc:.4f}")

# 9. Sauvegarde
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # remonte depuis scripts/ vers emotion_project/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Création du dossier s'il n'existe pas
os.makedirs(MODELS_DIR, exist_ok=True)

# Sauvegarde
model.save(os.path.join(MODELS_DIR, "emotion_model_improved.keras"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

print(f"\n✅ Modèle et encoder sauvegardés dans : {MODELS_DIR}")