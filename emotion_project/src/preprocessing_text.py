import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
import nlpaug.augmenter.word as naw
import nltk
from sklearn import preprocessing

df = pd.read_csv("texts/emotion_dataset_raw.csv")

print(dir(nfx))

# Nettoyage du texte dans une nouvelle colonne Clean_Text

#Supprimer les handles:utilisateurs (@username) dans le texte
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

#Supprimer les stopwords:  les mots inutiles("le", "and", "de", ...) dans le texte
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

stemmer = SnowballStemmer("english")
df['Clean_Text'] = (
    df['Clean_Text']
    .apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
)

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

print("Taille totale du dataset :", len(df))
print("Valeurs uniques dans la colonne 'Emotion' :", df['Emotion'].unique())
print("Nombre de valeurs manquantes :", df['Emotion'].isnull().sum())
print("Distribution des classes :\n", df['Emotion'].value_counts())


# Supprimer les lignes où 'text' ou 'emotion' est vide (NaN)
df_cleaned = df.dropna(subset=['Emotion', 'Text'])

#Supprimer les lignes où le texte est vide (chaîne vide "")
df_cleaned = df_cleaned[df_cleaned['Text'].str.strip() != '']

# Supprimer les doublons exacts (même texte et même label)
before = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates(subset=['Emotion', 'Text'])
print(f"Lignes supprimées (doublons) : {before - len(df_cleaned)}")

# garder juste les colonnes Emotion et Clean_Text
df_cleaned = df_cleaned[['Emotion', 'Clean_Text']]

# Supprimer les lignes où Clean_Text est vide après nettoyage
before = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=['Clean_Text'])
df_cleaned = df_cleaned[df_cleaned['Clean_Text'].str.strip() != '']
print(f"Lignes supprimées (Clean_Text vide ou NaN) : {before - len(df_cleaned)}")

print(f"Taille finale du dataset nettoyé : {len(df_cleaned)}")

#Data augmentation
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
minority_classes = ['disgust', 'shame', 'neutral']
def augment_minority_synonyms(df, minority_classes, n_aug=2):
    import nlpaug.augmenter.word as naw
    aug_syn = naw.SynonymAug(aug_src='wordnet')

    augmented = []
    for _, row in df.iterrows():
        if row['Emotion'] in minority_classes:
            augmented_texts = [row['Clean_Text']]
            for _ in range(n_aug):
                augmented_texts.append(aug_syn.augment(row['Clean_Text']))
            for text in augmented_texts:
                augmented.append({
                    'Emotion': row['Emotion'],
                    'Clean_Text': text
                })
        else:
            # convertir en dict pour uniformiser
            augmented.append({
                'Emotion': row['Emotion'],
                'Clean_Text': row['Clean_Text']
            })
    return pd.DataFrame(augmented)

df_cleaned_aug = augment_minority_synonyms(df_cleaned, minority_classes, n_aug=3)
print(f"Taille avant augmentation : {len(df_cleaned)}")
print(f"Taille après augmentation : {len(df_cleaned_aug)}")

# codage des emotions: 0 pour joy, .....
def fit_label_encoder(df):
    label_encoder = preprocessing.LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df)
    return label_encoder, encoded_labels

label_encoder, encoded_labels = fit_label_encoder(df_cleaned_aug['Emotion'])
df_cleaned_aug['N_label'] = encoded_labels


df_cleaned_aug['Clean_Text'] = df_cleaned_aug['Clean_Text'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
)


# Sauvegarde dans le dossier texts/
df_cleaned_aug.to_csv('texts/cleaned_emotion_dataset.csv', index=False) 

"""

# 1. Division en train (70%) et temp (30%)
df_train, df_temp = train_test_split(
    df_cleaned, test_size=0.30, random_state=42, stratify=df_cleaned['Emotion']
)

# 2. Division de temp en validation (15%) et test (15%)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, stratify=df_temp['Emotion']
)

print(df_train.head(10))

# 3. Sauvegarde dans le dossier texts/
df_train.to_csv('texts/train.csv', index=False)
df_val.to_csv('texts/val.csv', index=False)
df_test.to_csv('texts/test.csv', index=False)


print("Données enregistrées")

"""