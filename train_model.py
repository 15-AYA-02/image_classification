import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from feature_extraction import FeatureExtractor
from tqdm import tqdm

class ImageClassifier:
    """
    Classe pour entraîner et évaluer les classifieurs
    """
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.feature_extractor = FeatureExtractor()
        self.classifiers = {}
        self.best_classifier = None
        self.classes = []
        
    def load_dataset(self):
        """
        Charge toutes les images du dataset et extrait leurs features
        
        Returns:
            X: Features extraites (numpy array)
            y: Labels (numpy array)
        """
        print("📂 Chargement du dataset...")
        
        X = []  # Features
        y = []  # Labels
        
        # Parcourir chaque catégorie
        self.classes = sorted(os.listdir(self.dataset_path))
        self.classes = [c for c in self.classes if os.path.isdir(os.path.join(self.dataset_path, c))]
        
        print(f"📋 Catégories trouvées : {self.classes}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, class_name)
            
            # Lister toutes les images de cette catégorie
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   → {class_name}: {len(image_files)} images")
            
            # Extraire les features pour chaque image
            for img_file in tqdm(image_files, desc=f"Extraction {class_name}"):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Charger et prétraiter l'image
                    image = self.feature_extractor.load_and_preprocess_image(img_path)
                    
                    # Extraire les features
                    features = self.feature_extractor.extract_all_features(image)
                    
                    X.append(features)
                    y.append(class_idx)
                    
                except Exception as e:
                    print(f"⚠️ Erreur avec {img_file}: {e}")
                    continue
        
        print(f"✅ Dataset chargé: {len(X)} images au total\n")
        
        return np.array(X), np.array(y)
    
    def train_classifiers(self, X_train, y_train):
        """
        Entraîne les 3 classifieurs
        """
        print("🎓 Entraînement des classifieurs...\n")
        
        # 1. KNN (K-Nearest Neighbors)
        print("1️⃣ KNN...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        self.classifiers['KNN'] = knn
        
        # 2. SVM (Support Vector Machine)
        print("2️⃣ SVM...")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        self.classifiers['SVM'] = svm
        
        # 3. Random Forest
        print("3️⃣ Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.classifiers['RandomForest'] = rf
        
        print("✅ Entraînement terminé!\n")
    
    def evaluate_classifiers(self, X_test, y_test):
        """
        Évalue tous les classifieurs et trouve le meilleur
        """
        print("📊 Évaluation des classifieurs...\n")
        
        best_accuracy = 0
        
        for name, clf in self.classifiers.items():
            # Prédictions
            y_pred = clf.predict(X_test)
            
            # Calculer l'accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{'='*50}")
            print(f"📈 {name}")
            print(f"{'='*50}")
            print(f"Accuracy: {accuracy*100:.2f}%\n")
            
            # Rapport de classification détaillé
            print("Rapport de classification:")
            print(classification_report(y_test, y_pred, target_names=self.classes))
            
            # Matrice de confusion
            print("Matrice de confusion:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print("\n")
            
            # Garder le meilleur classifieur
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_classifier = clf
                self.best_classifier_name = name
        
        print(f"🏆 Meilleur classifieur: {self.best_classifier_name} ({best_accuracy*100:.2f}%)\n")
    
    def save_model(self, output_dir='models'):
        """
        Sauvegarde le meilleur modèle
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_data = {
            'classifier': self.best_classifier,
            'classifier_name': self.best_classifier_name,
            'classes': self.classes
        }
        
        model_path = os.path.join(output_dir, 'classifier_model.pkl')
        joblib.dump(model_data, model_path)
        
        print(f"💾 Modèle sauvegardé: {model_path}")
    
    def train(self, test_size=0.2):
        """
        Pipeline complet d'entraînement
        """
        print("\n" + "="*60)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("="*60 + "\n")
        
        # 1. Charger le dataset
        X, y = self.load_dataset()
        
        # 2. Diviser en train/test (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Données d'entraînement: {len(X_train)} images")
        print(f"📊 Données de test: {len(X_test)} images\n")
        
        # 3. Entraîner les classifieurs
        self.train_classifiers(X_train, y_train)
        
        # 4. Évaluer les classifieurs
        self.evaluate_classifiers(X_test, y_test)
        
        # 5. Sauvegarder le meilleur modèle
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("="*60 + "\n")


# Script principal
if __name__ == "__main__":
    # Créer et entraîner le classifieur
    classifier = ImageClassifier(dataset_path='dataset')
    classifier.train(test_size=0.2)