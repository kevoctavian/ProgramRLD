"""
ML Pipeline untuk Klasifikasi Penyakit Daun Padi
Menggunakan Random Forest + GLCM + Color Features + LBP + SMOTE
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
from PIL import Image


class ImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        # Simpan juga versi RGB untuk ekstraksi fitur warna
        self._last_bgr = resized
        return normalized

    def preprocess_pil(self, pil_image):
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        self._last_bgr = resized
        return normalized


class GLCMFeatureExtractor:
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles
        self.feature_names = [
            'contrast', 'dissimilarity', 'homogeneity',
            'energy', 'correlation', 'ASM'
        ]

    def extract_features(self, image):
        start_time = time.time()

        if image.dtype != np.uint8:
            image = img_as_ubyte(image)

        glcm = graycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {}
        angle_names = ['0', '45', '90', '135']

        for idx, angle_name in enumerate(angle_names):
            features[f'contrast_{angle_name}'] = graycoprops(glcm, 'contrast')[0, idx]
            features[f'dissimilarity_{angle_name}'] = graycoprops(glcm, 'dissimilarity')[0, idx]
            features[f'homogeneity_{angle_name}'] = graycoprops(glcm, 'homogeneity')[0, idx]
            features[f'energy_{angle_name}'] = graycoprops(glcm, 'energy')[0, idx]
            features[f'correlation_{angle_name}'] = graycoprops(glcm, 'correlation')[0, idx]
            features[f'ASM_{angle_name}'] = graycoprops(glcm, 'ASM')[0, idx]

        for prop in self.feature_names:
            values = [features[f'{prop}_{angle}'] for angle in angle_names]
            features[f'{prop}_mean'] = np.mean(values)

        feature_vector = []
        for angle in angle_names:
            for prop in self.feature_names:
                feature_vector.append(features[f'{prop}_{angle}'])

        extraction_time = time.time() - start_time
        features['extraction_time'] = extraction_time

        return features, np.array(feature_vector)

    def get_feature_names_ordered(self):
        names = []
        angle_names = ['0', '45', '90', '135']
        for angle in angle_names:
            for prop in self.feature_names:
                names.append(f'{prop}_{angle}')
        return names


# ========== FITUR BARU: COLOR FEATURES ==========
class ColorFeatureExtractor:
    """
    Ekstraksi fitur warna dari gambar BGR
    HSV + LAB color space — sangat efektif membedakan penyakit padi
    karena setiap penyakit punya warna khas (kuning, coklat, hijau)
    """

    def extract_features(self, bgr_image):
        features = {}

        # === HSV Color Space ===
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            ch = hsv[:, :, i].flatten()
            features[f'hsv_{channel}_mean'] = np.mean(ch)
            features[f'hsv_{channel}_std'] = np.std(ch)
            features[f'hsv_{channel}_skew'] = float(np.mean(((ch - np.mean(ch)) / (np.std(ch) + 1e-8)) ** 3))

        # === LAB Color Space ===
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        for i, channel in enumerate(['L', 'A', 'B']):
            ch = lab[:, :, i].flatten()
            features[f'lab_{channel}_mean'] = np.mean(ch)
            features[f'lab_{channel}_std'] = np.std(ch)

        # === Histogram BGR (8 bins per channel) ===
        for i, channel in enumerate(['B', 'G', 'R']):
            hist = cv2.calcHist([bgr_image], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()
            for j, val in enumerate(hist):
                features[f'hist_{channel}_{j}'] = val

        feature_vector = list(features.values())
        return features, np.array(feature_vector)

    def get_feature_count(self):
        # HSV: 3 channel x 3 stats = 9
        # LAB: 3 channel x 2 stats = 6
        # Histogram: 3 channel x 8 bins = 24
        return 39


# ========== FITUR BARU: LBP FEATURES ==========
class LBPFeatureExtractor:
    """
    Local Binary Pattern — menangkap pola tekstur lokal
    Sangat baik untuk membedakan tekstur bercak penyakit
    """

    def __init__(self, radius=3, n_points=24):
        self.radius = radius
        self.n_points = n_points

    def extract_features(self, gray_image):
        features = {}

        lbp = local_binary_pattern(
            gray_image,
            self.n_points,
            self.radius,
            method='uniform'
        )

        # Histogram LBP (26 bins untuk uniform LBP dengan n_points=24)
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        for i, val in enumerate(hist):
            features[f'lbp_hist_{i}'] = val

        # Statistik LBP
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = -np.sum(
            hist * np.log2(hist + 1e-10)
        )

        feature_vector = list(features.values())
        return features, np.array(feature_vector)

    def get_feature_count(self):
        return self.n_points + 2 + 3  # histogram + statistik


# ========== COMBINED PIPELINE ==========
class RiceDiseasePipeline:
    """
    Complete ML Pipeline dengan GLCM + Color + LBP features
    Total fitur: 24 (GLCM) + 39 (Color) + 29 (LBP) = 92 fitur
    """

    def __init__(self, model_path=None):
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = GLCMFeatureExtractor()
        self.color_extractor = ColorFeatureExtractor()
        self.lbp_extractor = LBPFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.class_names = [
            'bacterial_blight',
            'rice_blast',
            'tungro',
            'healthy',
        ]
        self.class_name_mapping = {
            'Bacterial Blight': 'bacterial_blight',
            'Rice Blast': 'rice_blast',
            'Tungro': 'tungro',
            'Healthy': 'healthy',
        }

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _extract_all_features(self, gray_image, bgr_image):
        """
        Ekstrak semua fitur: GLCM + Color + LBP
        """
        # GLCM features (24 fitur)
        glcm_features, glcm_vector = self.feature_extractor.extract_features(gray_image)

        # Color features (39 fitur)
        color_features, color_vector = self.color_extractor.extract_features(bgr_image)

        # LBP features (29 fitur)
        lbp_features, lbp_vector = self.lbp_extractor.extract_features(gray_image)

        # Gabungkan semua fitur
        combined_vector = np.concatenate([glcm_vector, color_vector, lbp_vector])

        # Gabungkan semua dictionary fitur
        all_features = {}
        all_features.update(glcm_features)
        all_features.update(color_features)
        all_features.update(lbp_features)

        return all_features, combined_vector

    def predict_single_image(self, image_path_or_array):
        if self.model is None:
            raise ValueError("Model belum di-load atau di-train!")

        start_total = time.time()

        # 1. Preprocessing
        start_prep = time.time()
        if isinstance(image_path_or_array, str):
            gray_image = self.preprocessor.preprocess(image_path_or_array)
        elif isinstance(image_path_or_array, Image.Image):
            gray_image = self.preprocessor.preprocess_pil(image_path_or_array)
        else:
            gray_image = image_path_or_array

        bgr_image = self.preprocessor._last_bgr
        preprocessing_time = time.time() - start_prep

        # 2. Feature Extraction (GLCM + Color + LBP)
        start_feat = time.time()
        all_features, combined_vector = self._extract_all_features(gray_image, bgr_image)
        extraction_time = time.time() - start_feat

        # 3. Prediction
        start_pred = time.time()
        feature_vector_scaled = self.scaler.transform([combined_vector])
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        prediction_time = time.time() - start_pred

        total_time = time.time() - start_total

        # Map ke nama database
        raw_class_name = self.class_names[prediction]
        db_class_name = self.class_name_mapping.get(raw_class_name, raw_class_name)

        # Ambil hanya glcm_features untuk disimpan ke database (sesuai model lama)
        glcm_features_only = {k: v for k, v in all_features.items()
                              if any(k.startswith(p) for p in
                                     ['contrast_', 'dissimilarity_', 'homogeneity_',
                                      'energy_', 'correlation_', 'ASM_', 'extraction_time'])}

        result = {
            'predicted_class': db_class_name,
            'predicted_class_index': int(prediction),
            'confidence': float(probabilities[prediction] * 100),
            'all_probabilities': {
                self.class_name_mapping.get(self.class_names[i], self.class_names[i]): float(prob * 100)
                for i, prob in enumerate(probabilities)
            },
            'glcm_features': glcm_features_only,
            'preprocessing_time': preprocessing_time,
            'extraction_time': extraction_time,
            'prediction_time': prediction_time,
            'total_time': total_time
        }

        return result

    def save_model(self, save_dir, model_name='rice_disease_rf_model'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, f'{model_name}.joblib')
        joblib.dump(self.model, model_path)

        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            'class_names': self.class_names,
            'feature_names': self.feature_extractor.get_feature_names_ordered(),
            'n_features': 92  # 24 GLCM + 39 Color + 29 LBP
        }
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.joblib')
        joblib.dump(metadata, metadata_path)

        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Metadata saved: {metadata_path}")

        return model_path, scaler_path, metadata_path

    def load_model(self, model_dir, model_name='rice_disease_rf_model'):
        model_path = os.path.join(model_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        scaler_path = os.path.join(model_dir, f'{model_name}_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded: {scaler_path}")

        metadata_path = os.path.join(model_dir, f'{model_name}_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.class_names = metadata['class_names']
            print(f"Metadata loaded: {metadata_path}")

        print("Model ready for prediction!")

    def train(self, X_train, y_train, apply_smote=True, **rf_params):
        """Untuk kompatibilitas dengan kode lama"""
        X_train_scaled = self.scaler.fit_transform(X_train)

        if apply_smote:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train

        default_params = {
            'n_estimators': 100, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1, 'verbose': 1
        }
        default_params.update(rf_params)

        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train_resampled, y_train_resampled)


# ========== UTILITY FUNCTIONS ==========
def load_dataset_for_training(dataset_path):
    """
    Load dataset dengan ekstraksi fitur GLCM + Color + LBP
    """
    preprocessor = ImagePreprocessor()
    glcm_extractor = GLCMFeatureExtractor()
    color_extractor = ColorFeatureExtractor()
    lbp_extractor = LBPFeatureExtractor()

    X = []
    y = []
    image_paths = []

    class_names = ['bacterial_blight', 'rice_blast', 'tungro', 'healthy']

    print("Loading dataset dengan fitur GLCM + Color + LBP...")
    print(f"Total fitur per gambar: 24 (GLCM) + 39 (Color) + 29 (LBP) = 92 fitur")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue

        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Processing {class_name}: {len(image_files)} images...")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load BGR untuk color features
                bgr = cv2.imread(img_path)
                bgr = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)

                # Preprocessing untuk gray
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

                # Ekstrak semua fitur
                _, glcm_vec = glcm_extractor.extract_features(gray)
                _, color_vec = color_extractor.extract_features(bgr)
                _, lbp_vec = lbp_extractor.extract_features(gray)

                # Gabungkan
                combined = np.concatenate([glcm_vec, color_vec, lbp_vec])

                X.append(combined)
                y.append(class_idx)
                image_paths.append(img_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset loaded: {len(X)} samples, {X.shape[1]} fitur")

    return X, y, image_paths


if __name__ == "__main__":
    print("Rice Disease Classification Pipeline")
    print("Features: GLCM (24) + Color/HSV/LAB (39) + LBP (29) = 92 total")