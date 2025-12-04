"""Point d'entrée : entraîne et sauvegarde le modèle final."""
from src.train_model import train_and_save

if __name__ == "__main__":
    f1 = train_and_save()
    print(f"[OK] Modèle entraîné et sauvegardé dans models/model.joblib | F1_test = {f1:.3f}")
