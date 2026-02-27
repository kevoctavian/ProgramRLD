# ProgramRLD
Lulus Tepat Waktu dan Wisuda. AMINNN TUHAN YESUS!!!
# Rice Disease Classification System

Sistem klasifikasi penyakit daun padi menggunakan **Random Forest** dan **GLCM Feature Extraction** dengan **SMOTE** untuk handling imbalanced data.

## 🌾 Overview

Aplikasi web berbasis Django untuk mendeteksi 5 jenis kondisi daun padi:
1. **Bacterial Leaf Blight** - Penyakit bakteri
2. **Rice Blast** - Penyakit jamur
3. **Tungro** - Penyakit virus
4. **Healthy** - Daun sehat
5. **Rice** - Kontrol

## 🎯 Features

- ✅ Upload dan diagnosis gambar daun padi
- ✅ Ekstraksi 24 fitur GLCM (6 fitur × 4 sudut)
- ✅ Klasifikasi menggunakan Random Forest
- ✅ Confidence score untuk setiap prediksi
- ✅ Riwayat diagnosis dengan filter dan search
- ✅ Informasi lengkap tentang penyakit
- ✅ Dashboard statistik dengan visualisasi
- ✅ Admin panel untuk manajemen data
- ✅ Database PostgreSQL untuk production-ready

## 🛠️ Tech Stack

- **Backend**: Django 5.2, Python 3.13
- **Database**: PostgreSQL
- **ML Libraries**: scikit-learn, OpenCV, scikit-image, imbalanced-learn
- **Frontend**: Bootstrap 5, Chart.js
- **Deployment**: Django + Gunicorn/uWSGI

## 📦 Installation

### 1. Prerequisites

- Python 3.10+
- PostgreSQL
- pip & virtualenv

### 2. Clone & Setup

```bash
# Clone repository
git clone <repository-url>
cd ProgramRLD

# Create virtual environment
python -m venv Env
source Env/bin/activate  # Linux/Mac
Env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Create PostgreSQL database
psql -U postgres
CREATE DATABASE rice_disease_db;
GRANT ALL PRIVILEGES ON DATABASE rice_disease_db TO postgres;
GRANT ALL ON SCHEMA public TO postgres;
\q

# Update settings.py with your database credentials
```

### 4. Django Setup

```bash
# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Populate disease data
python manage.py shell < populate_diseases.py

# Run server
python manage.py runserver
```

## 🤖 Training Model

### Prepare Dataset

Struktur folder dataset:
```
dataset/
├── bacterial_blight/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── rice_blast/
│   └── ...
├── tungro/
│   └── ...
├── healthy/
│   └── ...
└── rice/
    └── ...
```

### Train Model

```bash
python train_model.py --dataset /path/to/dataset --output ./ml_models

# With custom parameters
python train_model.py \
    --dataset /path/to/dataset \
    --output ./ml_models \
    --n-estimators 200 \
    --test-size 0.2 \
    --random-seed 42
```

### Arguments

- `--dataset`: Path ke folder dataset (required)
- `--output`: Output directory untuk model (default: ./ml_models)
- `--test-size`: Proporsi data testing (default: 0.2)
- `--n-estimators`: Jumlah trees di Random Forest (default: 100)
- `--no-smote`: Disable SMOTE (tidak disarankan)
- `--random-seed`: Random seed (default: 42)

## 📊 Metodologi

### 1. Preprocessing
- Resizing: 256×256 pixels
- Grayscale conversion
- Gaussian Filter (noise reduction)
- Normalisasi

### 2. Feature Extraction (GLCM)
- 6 Fitur tekstur:
  - Contrast
  - Correlation
  - Energy
  - Homogeneity
  - Dissimilarity
  - ASM (Angular Second Moment)
- 4 Sudut: 0°, 45°, 90°, 135°
- **Total: 24 fitur**

### 3. Handling Imbalanced Data
- SMOTE (Synthetic Minority Over-sampling Technique)
- Menyeimbangkan distribusi kelas

### 4. Classification
- Random Forest Classifier
- Default: 100 trees
- Cross-validation ready

## 🗂️ Project Structure

```
ProgramRLD/
├── manage.py
├── train_model.py              # Training script
├── populate_diseases.py        # Initial data
├── requirements.txt
├── ProgramRLD/                 # Django settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── appsRLD/                    # Main application
│   ├── models.py               # Database models
│   ├── views.py                # Views/controllers
│   ├── forms.py                # Form validation
│   ├── urls.py                 # URL routing
│   ├── admin.py                # Admin customization
│   ├── ml_pipeline.py          # ML pipeline
│   ├── templates/              # HTML templates
│   │   └── appsRLD/
│   │       ├── base.html
│   │       ├── home.html
│   │       ├── upload.html
│   │       ├── result.html
│   │       ├── history.html
│   │       └── ...
│   └── migrations/
├── media/                      # Uploaded images
│   ├── rice_leaves/
│   └── disease_references/
├── ml_models/                  # Trained models
│   ├── rice_disease_rf_model.joblib
│   ├── rice_disease_rf_model_scaler.joblib
│   ├── rice_disease_rf_model_metadata.joblib
│   └── visualizations/
│       ├── confusion_matrix.png
│       ├── class_distribution.png
│       └── metrics.png
└── static/                     # Static files (CSS/JS)
```

## 📖 Usage

### Web Interface

1. **Upload Image**
   - Navigate to Upload page
   - Drag & drop atau browse gambar
   - Click "Analyze & Diagnose"

2. **View Result**
   - Lihat penyakit yang terdeteksi
   - Confidence score
   - GLCM features
   - Treatment recommendations
   - Berikan feedback

3. **History**
   - Lihat riwayat diagnosis
   - Filter by disease, date, confidence
   - Search by filename

4. **Disease Info**
   - Informasi lengkap setiap penyakit
   - Symptoms, causes, treatment

5. **Statistics**
   - Dashboard statistik
   - Visualisasi distribusi
   - Performance metrics

### Admin Panel

Access: `http://localhost:8000/admin`

Features:
- Manage disease categories
- View all diagnoses
- Monitor system statistics
- Training history
- Custom visualizations

## 🔧 Configuration

### settings.py

```python
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'rice_disease_db',
        'USER': 'postgres',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Media Files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# ML Models
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
```

## 📈 Model Performance

Expected metrics (with proper dataset):
- **Accuracy**: >90%
- **Precision**: >88%
- **Recall**: >87%
- **F1-Score**: >88%

*Actual performance depends on dataset quality and quantity*

## 🚀 Deployment

### Production Checklist

```bash
# 1. Set DEBUG = False in settings.py
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

# 2. Collect static files
python manage.py collectstatic

# 3. Use Gunicorn/uWSGI
gunicorn ProgramRLD.wsgi:application --bind 0.0.0.0:8000

# 4. Configure Nginx as reverse proxy
# 5. Setup SSL certificate
# 6. Configure PostgreSQL for production
# 7. Set up backup system
```

## 🐛 Troubleshooting

### Common Issues

**1. Model not loaded**
```bash
# Check if model exists
ls ml_models/

# Retrain if needed
python train_model.py --dataset /path/to/dataset
```

**2. Database permission error**
```sql
-- Run in PostgreSQL
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON DATABASE rice_disease_db TO postgres;
```

**3. Missing dependencies**
```bash
pip install --break-system-packages opencv-python scikit-image
```

## 📚 Dataset

**Source**: [Rice Leaf and Crop Disease Detection Dataset](https://data.mendeley.com/datasets/g7tcwvshff/1)
- Published: November 2024
- Total: 10,766 images
- Classes: 5
- Format: JPG/JPEG

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Mendeley Data for the dataset
- scikit-learn, OpenCV, scikit-image communities
- Django framework
- Bootstrap for UI components

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Built with ❤️ for Indonesian Agriculture**
