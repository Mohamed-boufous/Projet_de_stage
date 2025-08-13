# scripts/train_image_model.py
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# Importer les variables et fonctions des autres fichiers
from src.preprocessing_img import train_dataset, test_dataset, class_names, get_train_labels
from src.image_model import create_model

# --- 1. Paramètres du Modèle ---
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = len(class_names)

print("Création du modèle CNN optimisé...")
model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()

# --- 2. Calcul des Poids des Classes (Correction du déséquilibre) --- # <-- AJOUT IMPORTANT
print("\nCalcul des poids des classes pour gérer le déséquilibre...")
# On utilise la fonction ajoutée pour récupérer tous les labels
train_labels = get_train_labels(train_dataset)

# Calcul des poids avec scikit-learn
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Poids des classes :", class_weight_dict)

# --- 3. Compilation du Modèle ---
print("\nCompilation du modèle...")
model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'), # Métrique utile pour l'déséquilibre
        tf.keras.metrics.Recall(name='recall')       # Métrique utile pour l'déséquilibre
    ]
)

# --- 4. Callbacks ---
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# --- 5. Entraînement ---
print("\nLancement de l'entraînement...")
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict  # <--- PASSAGE DES POIDS AU MODÈLE
)

# --- 6. Sauvegarde ---
print("\nSauvegarde du modèle entraîné...")
model.save('models/custom_cnn_v3_balanced.keras') # Nouveau nom de modèle
print("Modèle sauvegardé avec succès.")