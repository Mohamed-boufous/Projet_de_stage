# scripts/train_image_model.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.preprocessing_img import train_dataset, test_dataset, class_names
from src.image_model import create_model

# Les paramètres pour le modèle personnalisé (grayscale, 48x48)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = len(class_names)

print("Création du modèle CNN personnalisé...")
model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()

# Compilation du modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configuration des Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Lancement de l'entraînement (une seule phase)
print("\nLancement de l'entraînement...")
history = model.fit(
    train_dataset,
    epochs=100, # Nombre maximum d'époques
    validation_data=test_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Sauvegarde du modèle final
print("\nSauvegarde du modèle entraîné...")
model.save('models/custom_cnn_v1.keras')
print("Modèle sauvegardé avec succès.")