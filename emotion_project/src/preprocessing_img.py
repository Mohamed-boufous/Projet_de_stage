# src/preprocessing_img.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Constantes
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# Chemins (à adapter si nécessaire)
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath('.') # Pour les notebooks interactifs

TRAIN_DIR = os.path.join(BASE_DIR, 'images', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'images', 'test')

# Création des datasets
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

class_names = train_dataset.class_names

# Augmentation de données
data_augmentation_layers = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, augment=False):
    """Prépare le dataset : normalisation et augmentation."""
    ds = ds.map(lambda x, y: (layers.Rescaling(1./255)(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

# AJOUT : Fonction pour extraire les labels pour le calcul des poids de classe
def get_train_labels(ds):
    """Extrait tous les labels du dataset d'entraînement."""
    labels = []
    # Itérer sur le dataset non batché pour récupérer les labels
    for _, label_batch in ds.unbatch().batch(1024):
        labels.append(np.argmax(label_batch.numpy(), axis=1))
    return np.concatenate(labels)

# Préparation des datasets
train_dataset = prepare(train_dataset, augment=True)
test_dataset = prepare(test_dataset, augment=False)