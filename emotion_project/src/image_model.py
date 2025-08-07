# src/image_model.py

from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    """
    Crée l'architecture du modèle CNN pour la reconnaissance des émotions.
    
    Args:
        input_shape (tuple): La forme des images d'entrée (hauteur, largeur, canaux).
        num_classes (int): Le nombre de classes d'émotions à prédire.

    Returns:
        tf.keras.Model: Le modèle Keras compilé.
    """
    model = models.Sequential([
        # --- Couches de Convolution (Extraction des caractéristiques) ---
        # Ces couches agissent comme des loupes qui scannent l'image pour trouver
        # des motifs (bords, coins, textures).
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # --- Couches de Classification (Prise de décision) ---
        # Aplatit les caractéristiques 2D en un simple vecteur 1D.
        layers.Flatten(),
        
        # Couche dense pour apprendre les combinaisons de caractéristiques.
        layers.Dense(128, activation='relu'),
        
        # Le Dropout désactive aléatoirement des neurones pendant l'entraînement
        # pour éviter que le modèle ne "mémorise" trop les images (overfitting).
        layers.Dropout(0.5),
        
        # La couche de sortie finale. Elle a autant de neurones que de classes.
        # 'softmax' transforme les scores en probabilités pour chaque émotion.
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model