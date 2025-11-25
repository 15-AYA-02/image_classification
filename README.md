# Image Classification System — HOG, LBP, Color Histograms & Random Forest

##  Description du Projet

Ce projet consiste à construire un système complet de **classification d’images**, depuis l’extraction des caractéristiques jusqu’à la prédiction via une interface Streamlit.

Nous utilisons **trois familles de descripteurs** :

-  **Couleurs** : Histogrammes RGB + HSV  
-  **Formes** : HOG (Histogram of Oriented Gradients)  
-  **Textures** : LBP (Local Binary Patterns)

Ces informations sont combinées pour créer un vecteur riche (≈4223 valeurs), utilisé pour entraîner un modèle machine learning.

Après comparaison de plusieurs classifieurs (SVM, KNN, RandomForest…), le **Random Forest** a obtenu les meilleurs résultats avec une **accuracy de 89.47%**.

---

# 🏗️ Architecture Globale

## 🔹 Phase 1 : Entraînement

1. Chargement du dataset  
2. Extraction des features : couleur + forme + texture  
3. Entraînement de plusieurs modèles  
4. Sélection du meilleur modèle  
5. Sauvegarde du fichier : `classifier_model.pkl`

##  Phase 2 : Prédiction

1. Upload d’une image via l’interface Streamlit  
2. Extraction automatique des mêmes descripteurs  
3. Prédiction via Random Forest  
4. Affichage : catégorie + confiance  

Schéma du pipeline :

Dataset → Extraction Features → Tests Modèles → Meilleur Modèle → classifier_model.pkl

Upload image → Extraction features → Prédiction → Résultat

yaml
Copier le code

---

# Feature Engineering — Détails Techniques

## 1. Couleurs (RGB + HSV)
- Histogrammes RGB (4096 valeurs)
- Histogrammes HSV (48 valeurs)
- Analyse de la distribution des couleurs

##  2. Formes (HOG)
- Détection des contours  
- Analyse des orientations  
- Représente la structure globale

##  3. Textures (LBP)
- Analyse des micro-textures  
- Motifs répétitifs  

##  Vecteur Final
Tous les descripteurs sont concaténés :

Couleurs + HOG + LBP = Vecteur final (~4223 valeurs)

yaml
Copier le code

---

#  Modélisation

Plusieurs modèles testés :  
- SVM  
- KNN  
- Random Forest ✔️ (meilleur)

Résultat :  
- **Accuracy globale : 89.47%**  
- Très bonne performance sur les catégories :  
  - Fruits  
  - Textures  

Modèle sauvegardé dans :

models/classifier_model.pkl

yaml
Copier le code

---

#  Guide d’Exécution (AXE 5)

##  Étape 1 — Préparation

Créer les dossiers requis :

dataset/
models/

nginx
Copier le code

Installer les dépendances :

pip install -r requirements.txt

yaml
Copier le code

💡 *Astuce : Comme installer les outils avant de commencer un chantier.*

---

##  Étape 2 — Entraîner le modèle

Lancer :

python train_model.py

yaml
Copier le code

Ce script :
- extrait les features  
- teste les modèles  
- choisit le meilleur  
- génère `classifier_model.pkl`  

Vous verrez les accuracies dans la console.

---

##  Étape 3 — Lancer l’interface

Exécuter :

streamlit run app.py

yaml
Copier le code

Puis ouvrir :

👉 http://localhost:8501

L’utilisateur peut uploader une image et obtenir :
- la catégorie prédite  
- la confiance  

 *Même sans connaissance en machine learning, on peut tester le système.*

---

#  Résumé des 3 étapes

1. Installer les dépendances  
2. Entraîner le modèle  
3. Lancer l’interface Streamlit  

Avec ces 3 actions, le système est totalement fonctionnel ✔️

---

# 🧩 Améliorations Possibles

- Augmenter la taille du dataset  
- Ajouter des descripteurs plus avancés (SIFT, ORB, SURF…)  
- Tester des modèles modernes (CNN : ResNet, VGG, MobileNet)  
- Optimiser les hyperparamètres  
- Ajouter un système d’augmentation de données  

---

# 📦 Structure du Projet

image_classification/
│
├── app.py
├── train_model.py
├── feature_extraction.py
├── requirements.txt
│
├── dataset/
│ └── ... images ...
│
├── models/
│ └── classifier_model.pkl
│
└── README.md

yaml
Copier le code

---

#  Conclusion

Ce projet démontre la construction d’un pipeline ML complet, intégrant :

- Extraction intelligente des features  
- Sélection optimale du modèle  
- Interface utilisateur Streamlit simple et moderne  

Prêt pour une démonstration académique ou un déploiement local !
