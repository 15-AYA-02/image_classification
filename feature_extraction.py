import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

class FeatureExtractor:
    """
    Classe pour extraire les descripteurs d'images
    """
    
    def __init__(self):
        # Paramètres pour LBP
        self.lbp_radius = 3
        self.lbp_points = 8 * self.lbp_radius
        
    def extract_color_histogram(self, image, color_space='RGB'):
        """
        Extrait l'histogramme de couleur
        
        Args:
            image: Image en format numpy array
            color_space: 'RGB' ou 'HSV'
        
        Returns:
            Vecteur de features (histogramme aplati)
        """
        if color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculer l'histogramme pour chaque canal
        hist_features = []
        for i in range(3):  # 3 canaux (R,G,B ou H,S,V)
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = hist.flatten()
            # Normaliser
            hist = hist / (hist.sum() + 1e-7)
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def extract_hog(self, image):
        """
        Extrait les descripteurs HOG (Histogram of Oriented Gradients)
        
        Args:
            image: Image en format numpy array
        
        Returns:
            Vecteur de features HOG
        """
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Redimensionner pour avoir une taille fixe
        gray = cv2.resize(gray, (128, 128))
        
        # Extraire HOG
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        return features
    
    def extract_lbp(self, image):
        """
        Extrait les descripteurs LBP (Local Binary Patterns)
        
        Args:
            image: Image en format numpy array
        
        Returns:
            Histogramme LBP
        """
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Redimensionner
        gray = cv2.resize(gray, (128, 128))
        
        # Calculer LBP
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        
        # Calculer l'histogramme LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def extract_all_features(self, image):
        """
        Extrait TOUS les descripteurs et les combine
        
        Args:
            image: Image en format numpy array (RGB)
        
        Returns:
            Vecteur de features combiné
        """
        # 1. Histogramme RGB
        hist_rgb = self.extract_color_histogram(image, 'RGB')
        
        # 2. Histogramme HSV
        hist_hsv = self.extract_color_histogram(image, 'HSV')
        
        # 3. HOG
        hog_features = self.extract_hog(image)
        
        # 4. LBP
        lbp_features = self.extract_lbp(image)
        
        # Combiner tous les descripteurs
        combined_features = np.concatenate([
            hist_rgb,
            hist_hsv,
            hog_features,
            lbp_features
        ])
        
        return combined_features
    
    def load_and_preprocess_image(self, image_path):
        """
        Charge et prétraite une image
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Image prétraitée (numpy array RGB)
        """
        # Lire l'image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")
        
        # Convertir BGR vers RGB (OpenCV charge en BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner pour uniformiser
        image = cv2.resize(image, (256, 256))
        
        return image


# Exemple d'utilisation
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Tester sur une image
    image = extractor.load_and_preprocess_image("test_image.jpg")
    features = extractor.extract_all_features(image)
    
    print(f"Nombre de features extraites : {len(features)}")
    print(f"Features RGB+HSV : {96*2}")
    print(f"Features HOG : variable selon l'image")
    print(f"Features LBP : variable selon l'image")