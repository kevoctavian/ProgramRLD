# Rice Disease Classification System

Sistem klasifikasi penyakit daun padi menggunakan **Random Forest** dan **GLCM Feature Extraction** dengan **SMOTE** untuk handling imbalanced data.

## 🌾 Overview

Aplikasi web berbasis Django untuk mendeteksi 5 jenis kondisi daun padi:
1. **Bacterial Leaf Blight** - Penyakit bakteri
2. **Rice Blast** - Penyakit jamur
3. **Tungro** - Penyakit virus
4. **Healthy** - Daun sehat

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
