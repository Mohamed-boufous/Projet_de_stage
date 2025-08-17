# prepare_combined_dataset_DEFINITIVE.py
import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
PATH_FER2013 = "./datasets_source/fer2013"
PATH_CKPLUS_CSV = "./datasets_source/ckplus/ckextended.csv"
PATH_RAFDB_DATASET = "./datasets_source/raf-db/DATASET" # Le seul chemin nécessaire pour RAF-DB
PATH_OUTPUT = "./combined_dataset_96"
IMG_SIZE = 96

# --- Fonctions ---
def verify_paths():
    """Vérifie que les chemins sources existent."""
    print("Vérification des chemins d'accès...")
    paths_to_check = {
        "Dossier FER2013": PATH_FER2013, 
        "Fichier CK+ CSV": PATH_CKPLUS_CSV,
        "Dossier RAF-DB": PATH_RAFDB_DATASET
    }
    all_paths_ok = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"❌ ERREUR : {name} non trouvé à l'emplacement '{path}'")
            all_paths_ok = False
        else:
            print(f"✅ OK : {name} trouvé.")
    if not all_paths_ok: return False
    return True

def process_fer2013_from_folders(output_path):
    """Traite le dataset FER2013 depuis ses dossiers triés."""
    print("\nTraitement de FER2013...")
    for subset in ['train', 'test']:
        source_subset_dir = os.path.join(PATH_FER2013, subset)
        for emotion_folder in os.listdir(source_subset_dir):
            source_dir = os.path.join(source_subset_dir, emotion_folder)
            dest_dir = os.path.join(output_path, subset, emotion_folder)
            os.makedirs(dest_dir, exist_ok=True)
            for filename in os.listdir(source_dir):
                img = Image.open(os.path.join(source_dir, filename)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                img.save(os.path.join(dest_dir, f"fer_{filename}"))
    print("FER2013 terminé.")

def process_ckplus_from_csv(output_path):
    """Traite le dataset CK+ depuis son fichier CSV."""
    print("Traitement de CK+...")
    df = pd.read_csv(PATH_CKPLUS_CSV)
    ck_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"}
    for index, row in df.iterrows():
        label_str = ck_map.get(row['emotion'])
        if label_str is None or label_str == "contempt": continue
        dest_dir = os.path.join(output_path, 'train', label_str)
        os.makedirs(dest_dir, exist_ok=True)
        pixels = np.array(row['pixels'].split(), 'uint8')
        img = Image.fromarray(pixels.reshape(48, 48)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img.save(os.path.join(dest_dir, f"ck_{index}.png"))
    print("CK+ terminé.")

def process_rafdb_from_folders(output_path):
    """
    CORRECTION DÉFINITIVE : Traite RAF-DB depuis ses dossiers numérotés.
    """
    print("Traitement de RAF-DB...")
    raf_map = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}
    for subset in ['train', 'test']:
        source_subset_dir = os.path.join(PATH_RAFDB_DATASET, subset)
        if not os.path.exists(source_subset_dir): continue
        for emotion_folder_num in os.listdir(source_subset_dir):
            try:
                label_str = raf_map.get(int(emotion_folder_num))
                if label_str is None: continue
                source_dir = os.path.join(source_subset_dir, emotion_folder_num)
                dest_dir = os.path.join(output_path, subset, label_str)
                os.makedirs(dest_dir, exist_ok=True)
                for filename in os.listdir(source_dir):
                    img = Image.open(os.path.join(source_dir, filename)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                    img.save(os.path.join(dest_dir, f"raf_{filename}"))
            except (ValueError, FileNotFoundError):
                continue
    print("RAF-DB terminé.")


def verify_combination(output_path):
    """Vérifie que toutes les images ont bien été copiées."""
    print("\n--- Lancement de la Vérification Finale ---")
    count_fer = sum(len(files) for _, _, files in os.walk(PATH_FER2013))
    count_ck = len(pd.read_csv(PATH_CKPLUS_CSV).query("emotion != 7"))
    count_raf = sum(len(files) for _, _, files in os.walk(PATH_RAFDB_DATASET))
    
    dest_fer, dest_ck, dest_raf = 0, 0, 0
    for _, _, files in os.walk(output_path):
        for f in files:
            if f.startswith('fer_'): dest_fer += 1
            elif f.startswith('ck_'): dest_ck += 1
            elif f.startswith('raf_'): dest_raf += 1
    
    print("\n--- RAPPORT DE VÉRIFICATION ---")
    print(f"FER2013: {count_fer} images source -> {dest_fer} images combinées. {'✅ Succès' if count_fer == dest_fer else '❌ ÉCHEC'}")
    print(f"CK+ (sans 'contempt'): {count_ck} images source -> {dest_ck} images combinées. {'✅ Succès' if count_ck == dest_ck else '❌ ÉCHEC'}")
    print(f"RAF-DB: {count_raf} images source -> {dest_raf} images combinées. {'✅ Succès' if count_raf == dest_raf else '❌ ÉCHEC'}")
    
    total_dest = dest_fer + dest_ck + dest_raf
    if dest_fer > 30000 and dest_ck > 800 and dest_raf > 15000:
        print(f"\nLa combinaison des datasets a RÉUSSI ! Total de {total_dest} images.")
    else:
        print("\nLa combinaison des datasets a ÉCHOUÉ. Un des datasets n'a pas été copié.")

# --- SCRIPT PRINCIPAL ---
if __name__ == '__main__':
    if verify_paths():
        shutil.rmtree(PATH_OUTPUT, ignore_errors=True)
        os.makedirs(os.path.join(PATH_OUTPUT, 'train'), exist_ok=True)
        os.makedirs(os.path.join(PATH_OUTPUT, 'test'), exist_ok=True)
        
        process_fer2013_from_folders(PATH_OUTPUT)
        process_ckplus_from_csv(PATH_OUTPUT)
        process_rafdb_from_folders(PATH_OUTPUT)
        
        verify_combination(PATH_OUTPUT)