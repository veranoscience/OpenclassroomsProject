"""Point d'entrée simple : entraîne et sauvegarde le modèle """
from src.train_model import train_and_save

if __name__ == "__main__":
    f1 = train_and_save()
    print(f"Modèle entraîné et sauvegardé (models/model.joblib). F1 = {f1:.3f}")