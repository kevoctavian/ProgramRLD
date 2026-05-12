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

    def validate_rice_leaf(self, bgr_image):
            h, w = bgr_image.shape[:2]
            total_pixels = h * w
            gray      = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv       = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            score     = 0
            reasons   = []
    
            # ==============================================================
            # STEP 1
            # ==============================================================
            white_bg  = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 255]))
            gray_bg   = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 35, 200]))
            bg_mask   = cv2.bitwise_or(white_bg, gray_bg)
            non_bg    = cv2.bitwise_not(bg_mask)
            non_bg_px = max(cv2.countNonZero(non_bg), 1)
            bg_ratio  = cv2.countNonZero(bg_mask) / total_pixels
    
            # ==============================================================
            # STEP 2
            # ==============================================================
            leaf_aspect   = 0.0
            leaf_solidity = 0.0
            contours_fg, _ = cv2.findContours(non_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_fg:
                largest = max(contours_fg, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 100:
                    rect = cv2.minAreaRect(largest)
                    rw, rh = rect[1]
                    if min(rw, rh) > 0:
                        leaf_aspect = max(rw, rh) / min(rw, rh)
                    hull = cv2.convexHull(largest)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        leaf_solidity = area / hull_area
    
            # ==============================================================
            # STEP 3: Analisis warna DI AREA NON-BACKGROUND saja
            # ==============================================================
            sat_ch      = hsv[:, :, 1]
            sat_in_nonbg = float(np.mean(sat_ch[non_bg > 0])) if non_bg_px > 200 else 0
    
            def pct_nonbg(mask):
                return cv2.countNonZero(cv2.bitwise_and(mask, non_bg)) / non_bg_px
    
            green_m  = cv2.inRange(hsv, np.array([25, 15, 20]), np.array([90, 255, 255]))
            yellow_m = cv2.inRange(hsv, np.array([15, 15, 40]), np.array([35, 255, 255]))
            brown_m  = cv2.inRange(hsv, np.array([ 5, 15, 20]), np.array([25, 255, 220]))
    
            green_nb   = pct_nonbg(green_m)
            yellow_nb  = pct_nonbg(yellow_m)
            brown_nb   = pct_nonbg(brown_m)
            organic_nb = min(green_nb + yellow_nb + brown_nb, 1.0)
    
            # ==============================================================
            # KASUS A
            # ==============================================================
            is_leaf_on_white_bg = (
                bg_ratio > 0.40 and        # dominan background terang
                leaf_aspect > 2.5 and      # foreground elongated seperti daun
                sat_in_nonbg > 60 and      # foreground berwarna (bukan tulisan)
                organic_nb > 0.50          # warna hijau/kuning/coklat di foreground
            )
    
            if is_leaf_on_white_bg:
                return {
                    'is_valid':        True,
                    'score':           80,
                    'organic_ratio':   round(organic_nb * 100, 2),
                    'natural_green':   round(green_nb   * 100, 2),
                    'yellow_ratio':    round(yellow_nb  * 100, 2),
                    'brown_ratio':     round(brown_nb   * 100, 2),
                    'lbp_entropy':     0,
                    'local_std':       0,
                    'edge_ratio':      0,
                    'unnatural_ratio': 0,
                    'text_contours':   0,
                    'straight_lines':  0,
                    'non_bg_ratio':    round((1 - bg_ratio) * 100, 2),
                    'sat_mean':        round(sat_in_nonbg, 1),
                    'low_sat_ratio':   0,
                    'reasons':         [
                        f"daun padi terdeteksi di atas background terang "
                        f"(bg={bg_ratio*100:.0f}%, aspect={leaf_aspect:.1f}, "
                        f"sat_nonbg={sat_in_nonbg:.0f}, organic={organic_nb*100:.0f}%)"
                    ],
                }
    
            # ==============================================================
            # KASUS B
            # ==============================================================
            low_sat_total  = np.sum(hsv[:, :, 1] < 30) / total_pixels
            high_val_total = np.sum(hsv[:, :, 2] > 200) / total_pixels
            very_low_sat   = np.sum(hsv[:, :, 1] < 15)  / total_pixels
    
            is_document = (
                low_sat_total > 0.55 and
                high_val_total > 0.35 and
                leaf_aspect < 2.5      # tidak ada foreground elongated berwarna
            )
            is_grayscale = very_low_sat > 0.85
    
            if is_document:
                return self._invalid_result(
                    -60,
                    [f"terdeteksi sebagai kertas/dokumen "
                    f"(warna pudar {low_sat_total*100:.0f}%, "
                    f"cerah {high_val_total*100:.0f}%, "
                    f"tidak ada objek elongated)"]
                )
            if is_grayscale:
                return self._invalid_result(-50, ["gambar hitam-putih/grayscale tanpa warna daun"])
    
            # ==============================================================
            # KASUS C
            # ==============================================================
            hue_hist   = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            dom_hue    = float(hue_hist.max()) / total_pixels
            mser       = cv2.MSER_create()
            mser.setMinArea(20)
            mser.setMaxArea(800)
            regions, _ = mser.detectRegions(gray_blur)
            mser_count = len(regions)
    
            green_full = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([100, 255, 255]))
            green_cov  = cv2.countNonZero(green_full) / total_pixels
            g_sol, g_asp = 0.0, 0.0
            if green_cov > 0.3:
                ctrs_g, _ = cv2.findContours(green_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if ctrs_g:
                    lg = max(ctrs_g, key=cv2.contourArea)
                    if cv2.contourArea(lg) > 500:
                        hull_g   = cv2.convexHull(lg)
                        hull_a   = cv2.contourArea(hull_g)
                        if hull_a > 0: g_sol = cv2.contourArea(lg) / hull_a
                        rr       = cv2.minAreaRect(lg)
                        rw2, rh2 = rr[1]
                        if min(rw2, rh2) > 0: g_asp = max(rw2, rh2) / min(rw2, rh2)
    
            is_poster = dom_hue > 0.35 and (mser_count > 150 or (g_sol > 0.95 and g_asp < 1.5))
            if is_poster:
                return self._invalid_result(
                    -55,
                    [f"terdeteksi sebagai poster/desain grafis "
                    f"(warna dominan {dom_hue*100:.0f}%, teks={mser_count})"]
                )
    
            # ==============================================================
            # KASUS D
            # ==============================================================
            orange_m2 = cv2.inRange(hsv, np.array([  8, 100, 100]), np.array([ 28, 255, 255]))
            teal_m    = cv2.inRange(hsv, np.array([ 80,  80,  80]), np.array([110, 255, 220]))
            r1m       = cv2.inRange(hsv, np.array([  0, 120, 120]), np.array([  8, 255, 255]))
            r2m       = cv2.inRange(hsv, np.array([172, 120, 120]), np.array([180, 255, 255]))
            blue_m2   = cv2.inRange(hsv, np.array([110,  80,  80]), np.array([140, 255, 255]))
            animal_r  = (
                cv2.countNonZero(orange_m2) + cv2.countNonZero(teal_m) +
                cv2.countNonZero(r1m)       + cv2.countNonZero(r2m)   +
                cv2.countNonZero(blue_m2)
            ) / total_pixels
    
            g_not_leaf = green_cov > 0.3 and g_sol > 0.90 and g_asp < 1.8
            if animal_r > 0.04 and g_not_leaf:
                return self._invalid_result(
                    -50,
                    [f"terdeteksi subjek non-daun padi "
                    f"(warna hewan {animal_r*100:.1f}%, "
                    f"bentuk hijau tidak elongated: aspect={g_asp:.2f})"]
                )
            if animal_r > 0.12:
                return self._invalid_result(
                    -45,
                    [f"terdeteksi warna tubuh hewan dominan "
                    f"(orange/teal/merah/biru jenuh: {animal_r*100:.1f}%)"]
                )
    
            # ==============================================================
            # KASUS E
            # ==============================================================
            green_total   = cv2.countNonZero(green_m)  / total_pixels
            yellow_total  = cv2.countNonZero(yellow_m) / total_pixels
            brown_total   = cv2.countNonZero(brown_m)  / total_pixels
            organic_total = min(green_total + yellow_total + brown_total, 1.0)
    
            if   organic_total >= 0.55: score += 45
            elif organic_total >= 0.35: score += 35
            elif organic_total >= 0.20: score += 22
            elif organic_total >= 0.10: score += 12
            else:
                score -= 20
                reasons.append(f"warna organik rendah ({organic_total*100:.1f}%)")
    
            if green_total  >= 0.10: score += 15
            elif green_total  >= 0.04: score += 8
            if yellow_total >= 0.05: score += 8
            if brown_total  >= 0.05: score += 5
    
            lbp      = local_binary_pattern(gray_blur, 24, 3, method='uniform')
            hist_lbp, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
            lbp_entropy  = -np.sum(hist_lbp * np.log2(hist_lbp + 1e-10))
            local_std    = float(np.std(np.abs(cv2.Laplacian(gray_blur, cv2.CV_64F))))
            edges        = cv2.Canny(gray_blur, 20, 80)
            edge_ratio   = cv2.countNonZero(edges) / total_pixels
    
            if   lbp_entropy >= 3.8: score += 20
            elif lbp_entropy >= 2.8: score += 12
            elif lbp_entropy >= 1.8: score += 5
            else:
                score -= 30
                reasons.append(f"tekstur terlalu seragam (entropy={lbp_entropy:.2f})")
    
            if   local_std >= 5: score += 15
            elif local_std >= 2: score += 8
            if   edge_ratio >= 0.01: score += 8
    
            if mser_count > 200:
                score -= 40
                reasons.append(f"terdeteksi banyak teks ({mser_count} region MSER)")
            elif mser_count > 100:
                score -= 20
                reasons.append(f"terdeteksi teks ({mser_count} region MSER)")
    
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80, minLineLength=w * 0.30, maxLineGap=15
            )
            straight_lines = len(lines) if lines is not None else 0
            if straight_lines > 8:
                score -= 15
                reasons.append(f"terlalu banyak garis lurus ({straight_lines})")
    
            sat_mean = float(np.mean(hsv[:, :, 1]))
            is_valid = score >= 50
    
            if not is_valid and not reasons:
                reasons.append(f"kombinasi fitur tidak sesuai daun padi (score={score})")
    
            return {
                'is_valid':        is_valid,
                'score':           score,
                'organic_ratio':   round(organic_total   * 100, 2),
                'natural_green':   round(green_total      * 100, 2),
                'yellow_ratio':    round(yellow_total     * 100, 2),
                'brown_ratio':     round(brown_total      * 100, 2),
                'lbp_entropy':     round(lbp_entropy,     2),
                'local_std':       round(local_std,       2),
                'edge_ratio':      round(edge_ratio       * 100, 2),
                'unnatural_ratio': round(animal_r         * 100, 2),
                'text_contours':   mser_count,
                'straight_lines':  straight_lines,
                'non_bg_ratio':    round((1 - bg_ratio)   * 100, 2),
                'sat_mean':        round(sat_mean,         1),
                'low_sat_ratio':   round(low_sat_total     * 100, 2),
                'reasons':         reasons,
            }

    def _extract_all_features(self, gray_image, bgr_image):
        """
        Ekstrak semua fitur: GLCM + Color + LBP
        """
        glcm_features,  glcm_vector  = self.feature_extractor.extract_features(gray_image)
        color_features, color_vector = self.color_extractor.extract_features(bgr_image)
        lbp_features,   lbp_vector   = self.lbp_extractor.extract_features(gray_image)

        combined_vector = np.concatenate([glcm_vector, color_vector, lbp_vector])

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

        # 2. Validasi gambar daun padi
        validation = self.validate_rice_leaf(bgr_image)
        if not validation['is_valid']:
            return {
                'is_valid_leaf':         False,
                'predicted_class':       None,
                'predicted_class_index': -1,
                'confidence':            0.0,
                'all_probabilities':     {},
                'glcm_features':         {},
                'preprocessing_time':    preprocessing_time,
                'extraction_time':       0,
                'prediction_time':       0,
                'total_time':            time.time() - start_total,
                'validation':            validation,
                'rejection_reason': (
                    f"Gambar tidak terdeteksi sebagai daun padi. "
                    f"Warna organik terdeteksi hanya {validation['organic_ratio']}% "
                    f"(minimal 15%). Pastikan foto berisi daun padi yang jelas."
                )
            }

        # 3. Feature Extraction
        start_feat = time.time()
        all_features, combined_vector = self._extract_all_features(gray_image, bgr_image)
        extraction_time = time.time() - start_feat

        # 4. Prediction
        start_pred = time.time()
        feature_vector_scaled = self.scaler.transform([combined_vector])
        prediction    = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        prediction_time = time.time() - start_pred

        total_time = time.time() - start_total

        raw_class_name = self.class_names[prediction]
        db_class_name  = self.class_name_mapping.get(raw_class_name, raw_class_name)

        glcm_features_only = {k: v for k, v in all_features.items()
                              if any(k.startswith(p) for p in
                                     ['contrast_', 'dissimilarity_', 'homogeneity_',
                                      'energy_', 'correlation_', 'ASM_', 'extraction_time'])}

        return {
            'is_valid_leaf':         True,
            'predicted_class':       db_class_name,
            'predicted_class_index': int(prediction),
            'confidence':            float(probabilities[prediction] * 100),
            'all_probabilities': {
                self.class_name_mapping.get(self.class_names[i], self.class_names[i]): float(prob * 100)
                for i, prob in enumerate(probabilities)
            },
            'glcm_features':      glcm_features_only,
            'preprocessing_time': preprocessing_time,
            'extraction_time':    extraction_time,
            'prediction_time':    prediction_time,
            'total_time':         total_time,
            'validation':         validation,
        }

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
            'n_features': 92
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

# class RiceDiseasePipeline:
#     """
#     Complete ML Pipeline dengan GLCM + Color + LBP features
#     Total fitur: 24 (GLCM) + 39 (Color) + 29 (LBP) = 92 fitur
#     """

#     def __init__(self, model_path=None):
#         self.preprocessor = ImagePreprocessor()
#         self.feature_extractor = GLCMFeatureExtractor()
#         self.color_extractor = ColorFeatureExtractor()
#         self.lbp_extractor = LBPFeatureExtractor()
#         self.scaler = StandardScaler()
#         self.model = None
#         self.class_names = [
#             'bacterial_blight',
#             'rice_blast',
#             'tungro',
#             'healthy',
#         ]
#         self.class_name_mapping = {
#             'Bacterial Blight': 'bacterial_blight',
#             'Rice Blast': 'rice_blast',
#             'Tungro': 'tungro',
#             'Healthy': 'healthy',
#         }

#         if model_path and os.path.exists(model_path):
#             self.load_model(model_path)

#     def _extract_all_features(self, gray_image, bgr_image):
#         """
#         Ekstrak semua fitur: GLCM + Color + LBP
#         """
#         # GLCM features (24 fitur)
#         glcm_features, glcm_vector = self.feature_extractor.extract_features(gray_image)

#         # Color features (39 fitur)
#         color_features, color_vector = self.color_extractor.extract_features(bgr_image)

#         # LBP features (29 fitur)
#         lbp_features, lbp_vector = self.lbp_extractor.extract_features(gray_image)

#         # Gabungkan semua fitur
#         combined_vector = np.concatenate([glcm_vector, color_vector, lbp_vector])

#         # Gabungkan semua dictionary fitur
#         all_features = {}
#         all_features.update(glcm_features)
#         all_features.update(color_features)
#         all_features.update(lbp_features)

#         return all_features, combined_vector

#     def predict_single_image(self, image_path_or_array):
#         if self.model is None:
#             raise ValueError("Model belum di-load atau di-train!")

#         start_total = time.time()

#         # 1. Preprocessing
#         start_prep = time.time()
#         if isinstance(image_path_or_array, str):
#             gray_image = self.preprocessor.preprocess(image_path_or_array)
#         elif isinstance(image_path_or_array, Image.Image):
#             gray_image = self.preprocessor.preprocess_pil(image_path_or_array)
#         else:
#             gray_image = image_path_or_array

#         bgr_image = self.preprocessor._last_bgr
#         preprocessing_time = time.time() - start_prep

#         # 2. Feature Extraction (GLCM + Color + LBP)
#         start_feat = time.time()
#         all_features, combined_vector = self._extract_all_features(gray_image, bgr_image)
#         extraction_time = time.time() - start_feat

#         # 3. Prediction
#         start_pred = time.time()
#         feature_vector_scaled = self.scaler.transform([combined_vector])
#         prediction = self.model.predict(feature_vector_scaled)[0]
#         probabilities = self.model.predict_proba(feature_vector_scaled)[0]
#         prediction_time = time.time() - start_pred

#         total_time = time.time() - start_total

#         # Map ke nama database
#         raw_class_name = self.class_names[prediction]
#         db_class_name = self.class_name_mapping.get(raw_class_name, raw_class_name)

#         # Ambil hanya glcm_features untuk disimpan ke database (sesuai model lama)
#         glcm_features_only = {k: v for k, v in all_features.items()
#                               if any(k.startswith(p) for p in
#                                      ['contrast_', 'dissimilarity_', 'homogeneity_',
#                                       'energy_', 'correlation_', 'ASM_', 'extraction_time'])}

#         result = {
#             'predicted_class': db_class_name,
#             'predicted_class_index': int(prediction),
#             'confidence': float(probabilities[prediction] * 100),
#             'all_probabilities': {
#                 self.class_name_mapping.get(self.class_names[i], self.class_names[i]): float(prob * 100)
#                 for i, prob in enumerate(probabilities)
#             },
#             'glcm_features': glcm_features_only,
#             'preprocessing_time': preprocessing_time,
#             'extraction_time': extraction_time,
#             'prediction_time': prediction_time,
#             'total_time': total_time
#         }

#         return result

#     def save_model(self, save_dir, model_name='rice_disease_rf_model'):
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         model_path = os.path.join(save_dir, f'{model_name}.joblib')
#         joblib.dump(self.model, model_path)

#         scaler_path = os.path.join(save_dir, f'{model_name}_scaler.joblib')
#         joblib.dump(self.scaler, scaler_path)

#         metadata = {
#             'class_names': self.class_names,
#             'feature_names': self.feature_extractor.get_feature_names_ordered(),
#             'n_features': 92  # 24 GLCM + 39 Color + 29 LBP
#         }
#         metadata_path = os.path.join(save_dir, f'{model_name}_metadata.joblib')
#         joblib.dump(metadata, metadata_path)

#         print(f"✓ Model saved: {model_path}")
#         print(f"✓ Scaler saved: {scaler_path}")
#         print(f"✓ Metadata saved: {metadata_path}")

#         return model_path, scaler_path, metadata_path

#     def load_model(self, model_dir, model_name='rice_disease_rf_model'):
#         model_path = os.path.join(model_dir, f'{model_name}.joblib')
#         if os.path.exists(model_path):
#             self.model = joblib.load(model_path)
#             print(f"Model loaded: {model_path}")
#         else:
#             raise FileNotFoundError(f"Model not found: {model_path}")

#         scaler_path = os.path.join(model_dir, f'{model_name}_scaler.joblib')
#         if os.path.exists(scaler_path):
#             self.scaler = joblib.load(scaler_path)
#             print(f"Scaler loaded: {scaler_path}")

#         metadata_path = os.path.join(model_dir, f'{model_name}_metadata.joblib')
#         if os.path.exists(metadata_path):
#             metadata = joblib.load(metadata_path)
#             self.class_names = metadata['class_names']
#             print(f"Metadata loaded: {metadata_path}")

#         print("Model ready for prediction!")

#     def train(self, X_train, y_train, apply_smote=True, **rf_params):
#         """Untuk kompatibilitas dengan kode lama"""
#         X_train_scaled = self.scaler.fit_transform(X_train)

#         if apply_smote:
#             smote = SMOTE(random_state=42, k_neighbors=5)
#             X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
#         else:
#             X_train_resampled = X_train_scaled
#             y_train_resampled = y_train

#         default_params = {
#             'n_estimators': 100, 'max_depth': None,
#             'min_samples_split': 2, 'min_samples_leaf': 1,
#             'max_features': 'sqrt', 'random_state': 42,
#             'n_jobs': -1, 'verbose': 1
#         }
#         default_params.update(rf_params)

#         self.model = RandomForestClassifier(**default_params)
#         self.model.fit(X_train_resampled, y_train_resampled)


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