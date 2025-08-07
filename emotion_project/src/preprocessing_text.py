import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/texts/emotion_dataset_raw.csv")

print(dir(nfx))

#Supprimer les handles:utilisateurs (@username) dans le texte
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

#Supprimer les stopwords:  les mots inutiles("le", "and", "de", ...) dans le texte
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_punctuations)

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_urls)

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_hashtags)

df['Clean_Text'] = df['Clean_Text'].apply(lambda x: x.lower())

# Dictionnaire de normalisation des abréviations courantes
abbreviation_map = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "gr8": "great",
    "b4": "before",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "idk": "i do not know",
    "imo": "in my opinion",
    "l8r": "later",
    "btw": "by the way",
    "brb": "be right back",
    "ttyl": "talk to you later",
    "smh": "shaking my head",
    "fyi": "for your information",
    "jk": "just kidding",
    "lmao": "laughing my ass off",
    "nvm": "never mind",
    "np": "no problem",
    "omw": "on my way",
    "rofl": "rolling on the floor laughing",
    "tbh": "to be honest",
    "ty": "thank you",
    "yw": "you are welcome",
    "ily": "i love you",
    "ikr": "i know right",
    "idc": "i do not care",
    "hmu": "hit me up",
    "wyd": "what are you doing",
    "rn": "right now",
    "irl": "in real life",
    "asap": "as soon as possible",
    "bc": "because",
    "k": "okay",
    "tho": "though",
    "grats": "congratulations"
}

# Fonction pour remplacer les abréviations dans le texte
def normalize_abbreviations(text):
    words = text.split()
    normalized_words = [abbreviation_map.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Appliquer la normalisation sur la colonne Clean_Text
df['Clean_Text'] = df['Clean_Text'].apply(normalize_abbreviations)


#  Diviser en train et test (80% / 20%)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Emotion'])

#  Sauvegarder dans le dossier texts/
df_train.to_csv('data/texts/train.csv', index=False)
df_test.to_csv('data/texts/test.csv', index=False)

print("Données enregistrées")