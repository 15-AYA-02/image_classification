import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from feature_extraction import FeatureExtractor

# Configuration de la page
st.set_page_config(
    page_title="Classification d'images",
    page_icon="🖼️",
    layout="wide"
)

# Style CSS personnalisé (similaire à vos captures d'écran)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .upload-box {
        background-color: white;
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
    }
    .result-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🖼️ Classification d'images - Fruits vs Textures</h1>
        <p style='color: #666;'>Téléchargez une image pour la classifier automatiquement</p>
    </div>
""", unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        model_data = joblib.load('models/classifier_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("❌ Modèle non trouvé! Veuillez d'abord entraîner le modèle avec train_model.py")
        st.stop()

model_data = load_model()
classifier = model_data['classifier']
classes = model_data['classes']
classifier_name = model_data['classifier_name']

# Initialiser le feature extractor
feature_extractor = FeatureExtractor()

# Section d'upload
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div class='upload-box'>
            <h3>📤 Télécharger une image</h3>
            <p style='color: #666;'>Formats acceptés : JPG, JPEG, PNG</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez une image ici ou cliquez pour sélectionner",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Afficher l'image uploadée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_container_width=True)
        
        # Bouton pour charger une nouvelle image
        if st.button("📤 Télécharger une nouvelle image"):
            st.rerun()

with col2:
    if uploaded_file is not None:
        # Préparer l'image pour la prédiction
        with st.spinner("🔄 Analyse en cours..."):
            # Convertir PIL Image en numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Extraire les features
            features = feature_extractor.extract_all_features(img_array)
            features = features.reshape(1, -1)
            
            # Faire la prédiction
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            
            predicted_class = classes[prediction]
            confidence = probabilities[prediction] * 100
        
        # Afficher le résultat principal
        st.markdown(f"""
            <div class='result-box'>
                <h3 style='text-align: center;'>Catégorie prédite :</h3>
                <h1 style='text-align: center; color: #2e7d32;'>{predicted_class}</h1>
                <h3 style='text-align: center;'>Confiance : {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Barre de progression
        st.progress(confidence / 100)
        
        # Graphique des probabilités
        st.markdown("### 📊 Probabilités par catégorie")
        st.markdown("Distribution des scores de classification")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#7c4dff' if i == prediction else '#b39ddb' for i in range(len(classes))]
        bars = ax.bar(classes, probabilities * 100, color=colors)
        
        ax.set_ylabel('Probabilité (%)', fontsize=12)
        ax.set_xlabel('Catégories', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Informations supplémentaires
        with st.expander("ℹ️ Détails de la classification"):
            st.markdown(f"""
            - **Classifieur utilisé:** {classifier_name}
            - **Nombre de catégories:** {len(classes)}
            - **Catégories disponibles:** {', '.join(classes)}
            """)
            
            st.markdown("### 📈 Scores détaillés:")
            for i, class_name in enumerate(classes):
                st.write(f"**{class_name}:** {probabilities[i]*100:.2f}%")
    else:
        # Instructions quand aucune image n'est uploadée
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px;'>
                <h2 style='color: #666;'>👈 Commencez par télécharger une image</h2>
                <p>Les résultats de classification apparaîtront ici</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎓 Projet de Traitement et Analyse d'Images</p>
        <p>Méthodes: HOG, LBP, Histogrammes couleur + {classifier_name}</p>
    </div>
""".format(classifier_name=classifier_name), unsafe_allow_html=True)