# src/preprocessing_img.py

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# --- Définition des constantes ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64 # Taille des lots d'images à traiter

# --- Définition des chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'images', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'images', 'test')


# --- 1. Chargement des données ---
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- CORRECTION : On sauvegarde les noms des classes ICI ---
# On récupère les noms des classes avant que le dataset ne soit transformé.
class_names = train_dataset.class_names


# --- 2. Création des calques d'augmentation de données ---
data_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# --- 3. Préparation du pipeline de données ---
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
    # Applique la normalisation des pixels (rescale)
    ds = ds.map(lambda x, y: (layers.Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)
    
    # Applique l'augmentation uniquement sur le jeu de données d'entraînement
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    # Met en cache et pré-charge les données pour optimiser les performances
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_dataset = prepare(train_dataset, augment=True)
test_dataset = prepare(test_dataset, augment=False)


# --- Section principale pour tester le script ---
if __name__ == '__main__':
    print("Datasets créés avec la méthode moderne.")
    print("Classes détectées :", class_names)

    # Visualiser quelques images pour vérifier l'augmentation
    plt.figure(figsize=(12, 12))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            # On utilise la variable 'class_names' qu'on a sauvegardée
            class_name = class_names[labels[i].numpy().argmax()]
            plt.title(class_name)
            plt.axis("off")
    plt.suptitle("Exemples d'Images Augmentées (Méthode Moderne)")
    plt.show()