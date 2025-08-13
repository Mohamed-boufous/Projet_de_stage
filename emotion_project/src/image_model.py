# src/image_model.py
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape, num_classes):
    """Crée et retourne le modèle CNN."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Premier bloc convolutionnel
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('gelu'), # <-- MODIFICATION : 'gelu' est souvent plus performant que 'relu'
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # <-- MODIFICATION : Taux de dropout légèrement ajustés

        # Deuxième bloc convolutionnel
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Troisième bloc convolutionnel
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        # Quatrième bloc convolutionnel
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        
        # Couche de classification
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Activation('gelu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model