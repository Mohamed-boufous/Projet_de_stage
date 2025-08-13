import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df_cleaned_aug = pd.read_csv("texts/cleaned_emotion_dataset.csv") 

# Creating the Bag of Words model by applying Countvectorizer -convert textual data to numerical data
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))#example: the course was long-> [the,the course,the course was,course, course was, course was long,...]
data_cv = cv.fit_transform(df_cleaned_aug['Clean_Text']).toarray()

X_train, X_test, y_train, y_test =train_test_split(data_cv, df_cleaned_aug['N_label'], test_size=0.25, random_state=42)

# first neural network with keras tutorial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
# split into input (X) and output (y) variables
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='softmax'))
# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

from preprocessing_text import fit_label_encoder
label_encoder, df_cleaned_aug['N_label'] = fit_label_encoder(df_cleaned_aug['Emotion'])

text='I am happy'
array = cv.transform([text]).toarray()
pred = model.predict(array)
a=np.argmax(pred, axis=1)
print(label_encoder.inverse_transform(a)[0])


import tensorflow as tf
import pickle
import os

# Définir le chemin absolu/relatif vers le dossier models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Sauvegarde du modèle Keras
tf.keras.models.save_model(model, os.path.join(MODELS_DIR, 'my_model.h5'))

# Sauvegarde de l'encoder et du CountVectorizer
with open(os.path.join(MODELS_DIR, 'encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

with open(os.path.join(MODELS_DIR, 'CountVectorizer.pkl'), 'wb') as f:
    pickle.dump(cv, f)

"""
with open('encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
"""