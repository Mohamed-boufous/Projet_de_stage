# src/preprocessing_img.py
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Constantes
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# Chemins (inchangés)
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath('.')

TRAIN_DIR = os.path.join(BASE_DIR, 'images', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'images', 'test')

# --- MODIFICATIONS CI-DESSOUS ---

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',  # <-- MODIFICATION 1 : Passer de 'grayscale' à 'rgb'
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',  # <-- MODIFICATION 2 : Passer de 'grayscale' à 'rgb'
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation de données (peut rester la même)
data_augmentation_layers = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def prepare(ds, augment=False):
    """Prépare le dataset."""
    # <-- MODIFICATION 3 : La normalisation (Rescaling) est supprimée ici.
    # Le modèle EfficientNet a sa propre couche de normalisation.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

# Le reste du fichier (get_train_labels, etc.) reste identique.
# ... (gardez la fonction get_train_labels telle quelle)
def get_train_labels(ds):
    """Extrait tous les labels du dataset d'entraînement."""
    labels = []
    for _, label_batch in ds.unbatch().batch(1024):
        labels.append(np.argmax(label_batch.numpy(), axis=1))
    return np.concatenate(labels)

train_dataset = prepare(train_dataset, augment=True)
test_dataset = prepare(test_dataset, augment=False)