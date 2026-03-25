"""
ADVANCED Training Script - Target 90%+ Accuracy
Rice Disease Classification with Enhanced Features + Augmentation
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import cv2
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProgramRLD.settings')
import django
django.setup()

from appsRLD.ml_pipeline import (
    load_dataset_for_training,
    ImagePreprocessor,
    GLCMFeatureExtractor,
    ColorFeatureExtractor,
    LBPFeatureExtractor
)
from appsRLD.models import ModelTrainingHistory


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved")

def plot_roc_curve(y_test, y_score, classes, save_path):
    """
    Plot ROC Curve untuk multiclass (One vs Rest)
    """
    n_classes = len(classes)

    # Binarize label
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # Hitung ROC dan AUC per kelas
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr['micro'], tpr['micro'], _ = roc_curve(
        y_test_bin.ravel(), y_score.ravel()
    )
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot kiri: Per-class ROC ---
    colors = cycle(['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax = axes[0]

    for i, (color, class_name) in enumerate(zip(colors, classes)):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_name}\n(AUC = {roc_auc[i]:.4f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve per Kelas', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    # --- Plot kanan: Micro & Macro Average ---
    ax2 = axes[1]

    ax2.plot(
        fpr['micro'], tpr['micro'],
        color='#9b59b6', lw=2.5, linestyle='-',
        label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})'
    )
    ax2.plot(
        fpr['macro'], tpr['macro'],
        color='#1abc9c', lw=2.5, linestyle='--',
        label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})'
    )

    colors2 = cycle(['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    for i, (color, class_name) in enumerate(zip(colors2, classes)):
        ax2.plot(fpr[i], tpr[i], color=color, lw=1, alpha=0.4)

    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve - Micro & Macro Average', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    plt.suptitle(
        f'ROC Curve - Rice Disease Classification\n'
        f'Macro-AUC: {roc_auc["macro"]:.4f} | Micro-AUC: {roc_auc["micro"]:.4f}',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ ROC Curve saved")
    print(f"  Macro-AUC : {roc_auc['macro']:.4f}")
    print(f"  Micro-AUC : {roc_auc['micro']:.4f}")
    for i, class_name in enumerate(classes):
        print(f"  AUC {class_name:20s}: {roc_auc[i]:.4f}")

    return roc_auc

# ========== AUGMENTASI GAMBAR ==========
def augment_image_features(bgr_image, gray_image, glcm_ext, color_ext, lbp_ext):
    """
    Augmentasi gambar dan ekstrak fiturnya
    Menghasilkan beberapa variasi dari 1 gambar
    """
    augmented_vectors = []

    augmentations = [
        # Flip horizontal
        (cv2.flip(bgr_image, 1), cv2.flip(gray_image, 1)),
        # Flip vertical
        (cv2.flip(bgr_image, 0), cv2.flip(gray_image, 0)),
        # Rotate 90
        (cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE),
         cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)),
        # Rotate 180
        (cv2.rotate(bgr_image, cv2.ROTATE_180),
         cv2.rotate(gray_image, cv2.ROTATE_180)),
        # Brightness +30
        (cv2.convertScaleAbs(bgr_image, alpha=1.0, beta=30), gray_image),
        # Brightness -30
        (cv2.convertScaleAbs(bgr_image, alpha=1.0, beta=-30), gray_image),
    ]

    for bgr_aug, gray_aug in augmentations:
        try:
            _, glcm_vec = glcm_ext.extract_features(gray_aug)
            _, color_vec = color_ext.extract_features(bgr_aug)
            _, lbp_vec = lbp_ext.extract_features(gray_aug)
            combined = np.concatenate([glcm_vec, color_vec, lbp_vec])
            augmented_vectors.append(combined)
        except Exception:
            continue

    return augmented_vectors


def load_dataset_with_augmentation(dataset_path, augment_minority=True, augment_threshold=600):
    """
    Load dataset dengan augmentasi untuk kelas minoritas
    Kelas dengan gambar < augment_threshold akan diaugmentasi
    """
    preprocessor = ImagePreprocessor()
    glcm_extractor = GLCMFeatureExtractor()
    color_extractor = ColorFeatureExtractor()
    lbp_extractor = LBPFeatureExtractor()

    X = []
    y = []

    class_names = ['bacterial_blight', 'rice_blast', 'tungro', 'healthy']

    print("Loading dataset dengan augmentasi kelas minoritas...")
    print(f"Augmentasi aktif untuk kelas dengan < {augment_threshold} gambar")
    print(f"Total fitur: 24 (GLCM) + 39 (Color) + 29 (LBP) = 92 fitur\n")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue

        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        is_minority = len(image_files) < augment_threshold
        print(f"Processing {class_name}: {len(image_files)} images "
              f"{'[AUGMENTED]' if is_minority and augment_minority else ''}")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                bgr = cv2.imread(img_path)
                bgr = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

                # Fitur original
                _, glcm_vec = glcm_extractor.extract_features(gray)
                _, color_vec = color_extractor.extract_features(bgr)
                _, lbp_vec = lbp_extractor.extract_features(gray)
                combined = np.concatenate([glcm_vec, color_vec, lbp_vec])

                X.append(combined)
                y.append(class_idx)

                # Augmentasi hanya untuk kelas minoritas
                if augment_minority and is_minority:
                    aug_vectors = augment_image_features(
                        bgr, gray,
                        glcm_extractor, color_extractor, lbp_extractor
                    )
                    for aug_vec in aug_vectors:
                        X.append(aug_vec)
                        y.append(class_idx)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Hitung total setelah augmentasi
        total_class = sum(1 for label in y if label == class_idx)
        if is_minority and augment_minority:
            print(f"  → Setelah augmentasi: {total_class} samples")

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset total: {len(X)} samples, {X.shape[1]} fitur")

    return X, y


def train_model(dataset_path, output_dir, use_ensemble=True,
                smote_type='borderline', random_state=42):

    print("=" * 80)
    print("RICE DISEASE CLASSIFICATION")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    class_names = ['Bacterial Blight', 'Rice Blast', 'Tungro', 'Healthy']

    # ========== STEP 1: LOAD + AUGMENT ==========
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET + AUGMENTASI KELAS MINORITAS")
    print("=" * 80)

    X, y = load_dataset_with_augmentation(
        dataset_path,
        augment_minority=True,
        augment_threshold=700  # Kelas < 700 gambar akan diaugmentasi
    )

    # Sedikit noise
    X = X + np.random.normal(0, 0.005, X.shape)

    print(f"\n✓ Dataset setelah augmentasi:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  - {class_names[cls]:20s}: {count:5d} ({count/len(y)*100:.2f}%)")

    # ========== STEP 2: SPLIT ==========
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATASET (70/15/15)")
    print("=" * 80)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=random_state, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ========== STEP 3: PREPROCESSING ==========
    print("\n" + "=" * 80)
    print("STEP 3: SCALING + SMOTE")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Applying {smote_type.upper()} SMOTE...")
    print(f"  Before SMOTE: {len(X_train_scaled)} samples")

    if smote_type == 'borderline':
        smote = BorderlineSMOTE(random_state=random_state, k_neighbors=5, kind='borderline-1')
    elif smote_type == 'smoteenn':
        smote = SMOTEENN(random_state=random_state)
    else:
        smote = SMOTE(random_state=random_state, k_neighbors=3)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"  After SMOTE:  {len(X_train_resampled)} samples")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    - {class_names[cls]:20s}: {count}")

    # ========== STEP 4: TRAINING ==========
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING")
    print("=" * 80)

    training_start = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # --- Random Forest Grid Search ---
    print("\n🔧 Grid Search Random Forest...")
    rf_param_grid = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [None, 30, 40],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train_resampled, y_train_resampled)
    best_rf = rf_grid.best_estimator_
    print(f"  ✓ Best RF: {rf_grid.best_params_}")
    print(f"  ✓ Best CV Score: {rf_grid.best_score_*100:.2f}%")

    # --- Extra Trees --- (DINONAKTIFKAN - hanya RF)
    # print("\n🔧 Training Extra Trees...")
    # et = ExtraTreesClassifier(
    #     n_estimators=800,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features='sqrt',
    #     class_weight='balanced',
    #     random_state=random_state,
    #     n_jobs=-1
    # )
    # et.fit(X_train_resampled, y_train_resampled)
    # et_val_acc = accuracy_score(y_val, et.predict(X_val_scaled))
    # print(f"  ✓ Extra Trees Val Acc: {et_val_acc*100:.2f}%")

    # --- Gradient Boosting --- (DINONAKTIFKAN - hanya RF)
    # print("\n🔧 Training Gradient Boosting...")
    # gb = GradientBoostingClassifier(
    #     n_estimators=400,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     subsample=0.8,
    #     min_samples_split=3,
    #     random_state=random_state
    # )
    # gb.fit(X_train_resampled, y_train_resampled)
    # gb_val_acc = accuracy_score(y_val, gb.predict(X_val_scaled))
    # print(f"  ✓ Gradient Boosting Val Acc: {gb_val_acc*100:.2f}%")

    # --- Voting Ensemble --- (DINONAKTIFKAN - hanya RF)
    # print("\n🔧 Building Voting Ensemble...")
    # model = VotingClassifier(
    #     estimators=[
    #         ('rf', best_rf),
    #         ('et', et),
    #         ('gb', gb)
    #     ],
    #     voting='soft',
    #     n_jobs=-1
    # )
    # model.fit(X_train_resampled, y_train_resampled)

    # Gunakan Random Forest saja sebagai model final
    print("\n🔧 Menggunakan Random Forest sebagai model final...")
    model = best_rf
    print(f"  ✓ Model: Random Forest (best from GridSearch)")

    training_duration = time.time() - training_start
    print(f"\n✓ Training selesai dalam {training_duration:.2f}s")

    # ========== STEP 5: EVALUASI ==========
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATION")
    print("=" * 80)

    y_val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred) * 100

    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    test_precision = precision_score(y_test, y_test_pred, average='weighted') * 100
    test_recall = recall_score(y_test, y_test_pred, average='weighted') * 100
    test_f1 = f1_score(y_test, y_test_pred, average='weighted') * 100

    print(f"\n📊 Results:")
    print(f"  Validation Accuracy : {val_acc:.2f}%")
    print(f"  Testing Accuracy    : {test_acc:.2f}%")
    print(f"  Testing Precision   : {test_precision:.2f}%")
    print(f"  Testing Recall      : {test_recall:.2f}%")
    print(f"  Testing F1-Score    : {test_f1:.2f}%")

    print("\n" + "-" * 80)
    print("Classification Report:")
    print("-" * 80)
    print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_test_pred)
    print("Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        print(f"  - {class_name:20s}: {class_acc:.2f}%")

    plot_confusion_matrix(cm, class_names, os.path.join(viz_dir, 'confusion_matrix.png'))

    # ROC Curve
    print("\n📈 Generating ROC Curve...")
    y_score = model.predict_proba(X_test_scaled)
    roc_auc_scores = plot_roc_curve(
        y_test, y_score, class_names,
        os.path.join(viz_dir, 'roc_curve.png')
    )

    # ========== STEP 6: SAVE ==========
    print("\n" + "=" * 80)
    print("STEP 6: SAVING MODEL")
    print("=" * 80)

    model_path = os.path.join(output_dir, 'rice_disease_rf_model.joblib')
    joblib.dump(model, model_path)

    scaler_path = os.path.join(output_dir, 'rice_disease_rf_model_scaler.joblib')
    joblib.dump(scaler, scaler_path)

    metadata = {
        'class_names': class_names,
        'n_features': 92,
        'use_ensemble': use_ensemble,
        'smote_type': smote_type,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'roc_auc_macro': roc_auc_scores['macro'],
        'roc_auc_micro': roc_auc_scores['micro'],
        'roc_auc_per_class': {
            class_names[i]: roc_auc_scores[i]
            for i in range(len(class_names))
        }
    }
    metadata_path = os.path.join(output_dir, 'rice_disease_rf_model_metadata.joblib')
    joblib.dump(metadata, metadata_path)

    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Metadata saved: {metadata_path}")

    # ========== STEP 7: DATABASE ==========
    try:
        # Buat per-class metrics dari classification report
        report = classification_report(
            y_test, y_test_pred,
            target_names=class_names,
            output_dict=True
        )
        per_class = {
            class_name: {
                'precision': round(report[class_name]['precision'] * 100, 2),
                'recall':    round(report[class_name]['recall']    * 100, 2),
                'f1':        round(report[class_name]['f1-score']  * 100, 2),
                'support':   int(report[class_name]['support']),
            }
            for class_name in class_names
        }

        ModelTrainingHistory.objects.filter(is_active=True).update(is_active=False)
        ModelTrainingHistory.objects.create(
            model_name='Random Forest + Augmentasi',
            version=f'v{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            dataset_name='RLD_Dataset_Clean (4 Classes + Augmentasi)',
            total_samples=len(X),
            training_samples=len(X_train),
            validation_samples=len(X_val),
            test_samples=len(X_test),
            smote_applied=True,
            samples_after_smote=len(X_train_resampled),
            accuracy=test_acc,
            precision=test_precision,
            recall=test_recall,
            f1_score=test_f1,
            val_accuracy=val_acc,
            per_class_metrics=per_class,
            training_duration=training_duration / 60,
            model_file_path=model_path,
            is_active=True,
            notes=f'GLCM+Color+LBP+Augmentasi, SMOTE: {smote_type}, Val: {val_acc:.2f}%'
        )
        print("✓ Training history saved to database")
        print(f"  Per-class metrics saved: {list(per_class.keys())}")
    except Exception as e:
        print(f"⚠ Database save failed: {e}")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Validation : {val_acc:.2f}%")
    print(f"  Testing    : {test_acc:.2f}%")
    print(f"  F1-Score   : {test_f1:.2f}%")
    print(f"  Duration   : {training_duration:.2f}s")
    print("=" * 80)

    if test_acc >= 90:
        print("🎯 TARGET ACHIEVED: 90%+ ACCURACY!")
    elif test_acc >= 85:
        print("✅ EXCELLENT: 85%+ accuracy achieved!")
    else:
        print(f"📊 Current: {test_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='./ml_models')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--smote-type', type=str, default='borderline',
                        choices=['borderline', 'smoteenn', 'standard'])
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        use_ensemble=args.ensemble,
        smote_type=args.smote_type,
        random_state=args.random_seed
    )


if __name__ == '__main__':
    main()