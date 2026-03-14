from django.db import models

# Create your models here.
"""
Models untuk Aplikasi Klasifikasi Penyakit Daun Padi
Menggunakan Random Forest + GLCM + SMOTE
5 Kelas: Bacterial Leaf Blight, Rice Blast, Tungro, Healthy, Rice
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os


# ========== MODEL 1: DISEASE CATEGORIES ==========
class DiseaseCategory(models.Model):
    """
    Model untuk menyimpan kategori penyakit padi (4 kelas)
    """
    DISEASE_CHOICES = [
        ('bacterial_blight', 'Bacterial Leaf Blight'),
        ('rice_blast', 'Rice Blast'),
        ('tungro', 'Tungro'),
        ('healthy', 'Healthy'),
    ]
    
    name = models.CharField(
        max_length=50, 
        choices=DISEASE_CHOICES,
        unique=True,
        verbose_name="Nama Penyakit"
    )
    display_name = models.CharField(
        max_length=100,
        verbose_name="Nama Tampilan"
    )
    description = models.TextField(
        verbose_name="Deskripsi Penyakit",
        blank=True
    )
    symptoms = models.TextField(
        verbose_name="Gejala",
        help_text="Gejala yang terlihat pada daun padi",
        blank=True
    )
    causes = models.TextField(
        verbose_name="Penyebab",
        help_text="Penyebab penyakit (bakteri/jamur/virus)",
        blank=True
    )
    treatment = models.TextField(
        verbose_name="Penanganan",
        help_text="Cara menangani dan mencegah penyakit",
        blank=True
    )
    severity_level = models.IntegerField(
        verbose_name="Tingkat Keparahan",
        choices=[
            (1, 'Ringan'),
            (2, 'Sedang'),
            (3, 'Berat'),
            (0, 'Tidak Berlaku'),
        ],
        default=0
    )
    reference_image = models.ImageField(
        upload_to='disease_references/',
        verbose_name="Gambar Referensi",
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Kategori Penyakit"
        verbose_name_plural = "Kategori Penyakit"
        ordering = ['name']
    
    def __str__(self):
        return self.display_name
    
    def get_sample_count(self):
        """Hitung jumlah sampel diagnosis untuk kategori ini"""
        return self.diagnoses.count()


# ========== MODEL 2: UPLOADED IMAGES ==========
class RiceLeafImage(models.Model):
    """
    Model untuk menyimpan gambar daun padi yang diupload user
    """
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='uploaded_images',
        verbose_name="User",
        null=True,
        blank=True
    )
    image = models.ImageField(
        upload_to='rice_leaves/%Y/%m/%d/',
        verbose_name="Gambar Daun Padi"
    )
    original_filename = models.CharField(
        max_length=255,
        verbose_name="Nama File Asli"
    )
    file_size = models.IntegerField(
        verbose_name="Ukuran File (bytes)",
        blank=True,
        null=True
    )
    image_width = models.IntegerField(
        verbose_name="Lebar Gambar (px)",
        blank=True,
        null=True
    )
    image_height = models.IntegerField(
        verbose_name="Tinggi Gambar (px)",
        blank=True,
        null=True
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Waktu Upload"
    )
    ip_address = models.GenericIPAddressField(
        verbose_name="IP Address",
        blank=True,
        null=True
    )
    
    class Meta:
        verbose_name = "Gambar Daun Padi"
        verbose_name_plural = "Gambar Daun Padi"
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.original_filename} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_mb(self):
        """Konversi ukuran file ke MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0


# ========== MODEL 3: GLCM FEATURES ==========
class GLCMFeatures(models.Model):
    """
    Model untuk menyimpan hasil ekstraksi fitur GLCM
    """
    image = models.OneToOneField(
        RiceLeafImage,
        on_delete=models.CASCADE,
        related_name='glcm_features',
        verbose_name="Gambar"
    )
    
    # GLCM Features - 0°
    contrast_0 = models.FloatField(verbose_name="Contrast 0°", default=0)
    dissimilarity_0 = models.FloatField(verbose_name="Dissimilarity 0°", default=0)
    homogeneity_0 = models.FloatField(verbose_name="Homogeneity 0°", default=0)
    energy_0 = models.FloatField(verbose_name="Energy 0°", default=0)
    correlation_0 = models.FloatField(verbose_name="Correlation 0°", default=0)
    asm_0 = models.FloatField(verbose_name="ASM 0°", default=0)
    
    # GLCM Features - 45°
    contrast_45 = models.FloatField(verbose_name="Contrast 45°", default=0)
    dissimilarity_45 = models.FloatField(verbose_name="Dissimilarity 45°", default=0)
    homogeneity_45 = models.FloatField(verbose_name="Homogeneity 45°", default=0)
    energy_45 = models.FloatField(verbose_name="Energy 45°", default=0)
    correlation_45 = models.FloatField(verbose_name="Correlation 45°", default=0)
    asm_45 = models.FloatField(verbose_name="ASM 45°", default=0)
    
    # GLCM Features - 90°
    contrast_90 = models.FloatField(verbose_name="Contrast 90°", default=0)
    dissimilarity_90 = models.FloatField(verbose_name="Dissimilarity 90°", default=0)
    homogeneity_90 = models.FloatField(verbose_name="Homogeneity 90°", default=0)
    energy_90 = models.FloatField(verbose_name="Energy 90°", default=0)
    correlation_90 = models.FloatField(verbose_name="Correlation 90°", default=0)
    asm_90 = models.FloatField(verbose_name="ASM 90°", default=0)
    
    # GLCM Features - 135°
    contrast_135 = models.FloatField(verbose_name="Contrast 135°", default=0)
    dissimilarity_135 = models.FloatField(verbose_name="Dissimilarity 135°", default=0)
    homogeneity_135 = models.FloatField(verbose_name="Homogeneity 135°", default=0)
    energy_135 = models.FloatField(verbose_name="Energy 135°", default=0)
    correlation_135 = models.FloatField(verbose_name="Correlation 135°", default=0)
    asm_135 = models.FloatField(verbose_name="ASM 135°", default=0)
    
    # Average Features
    contrast_mean = models.FloatField(verbose_name="Contrast Mean", default=0)
    dissimilarity_mean = models.FloatField(verbose_name="Dissimilarity Mean", default=0)
    homogeneity_mean = models.FloatField(verbose_name="Homogeneity Mean", default=0)
    energy_mean = models.FloatField(verbose_name="Energy Mean", default=0)
    correlation_mean = models.FloatField(verbose_name="Correlation Mean", default=0)
    asm_mean = models.FloatField(verbose_name="ASM Mean", default=0)
    
    extraction_time = models.FloatField(
        verbose_name="Waktu Ekstraksi (detik)",
        help_text="Waktu yang dibutuhkan untuk ekstraksi fitur GLCM",
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Fitur GLCM"
        verbose_name_plural = "Fitur GLCM"
    
    def __str__(self):
        return f"GLCM Features - {self.image.original_filename}"
    
    def get_feature_vector(self):
        """Return semua fitur sebagai list untuk prediksi"""
        return [
            self.contrast_0, self.dissimilarity_0, self.homogeneity_0,
            self.energy_0, self.correlation_0, self.asm_0,
            self.contrast_45, self.dissimilarity_45, self.homogeneity_45,
            self.energy_45, self.correlation_45, self.asm_45,
            self.contrast_90, self.dissimilarity_90, self.homogeneity_90,
            self.energy_90, self.correlation_90, self.asm_90,
            self.contrast_135, self.dissimilarity_135, self.homogeneity_135,
            self.energy_135, self.correlation_135, self.asm_135,
        ]


# ========== MODEL 4: DIAGNOSIS RESULTS ==========
class DiagnosisResult(models.Model):
    """
    Model untuk menyimpan hasil diagnosis/prediksi
    """
    image = models.ForeignKey(
        RiceLeafImage,
        on_delete=models.CASCADE,
        related_name='diagnoses',
        verbose_name="Gambar"
    )
    predicted_disease = models.ForeignKey(
        DiseaseCategory,
        on_delete=models.SET_NULL,
        related_name='diagnoses',
        verbose_name="Penyakit Terdeteksi",
        null=True
    )
    
    # Confidence scores untuk setiap kelas
    confidence_bacterial_blight = models.FloatField(
        verbose_name="Confidence Bacterial Blight (%)",
        default=0
    )
    confidence_rice_blast = models.FloatField(
        verbose_name="Confidence Rice Blast (%)",
        default=0
    )
    confidence_tungro = models.FloatField(
        verbose_name="Confidence Tungro (%)",
        default=0
    )
    confidence_healthy = models.FloatField(
        verbose_name="Confidence Healthy (%)",
        default=0
    )
    
    max_confidence = models.FloatField(
        verbose_name="Confidence Tertinggi (%)",
        help_text="Tingkat keyakinan prediksi"
    )
    
    # Model Information
    model_name = models.CharField(
        max_length=100,
        verbose_name="Nama Model",
        default="Random Forest + GLCM"
    )
    model_version = models.CharField(
        max_length=50,
        verbose_name="Versi Model",
        default="v1.0"
    )
    
    # Processing Time
    preprocessing_time = models.FloatField(
        verbose_name="Waktu Preprocessing (detik)",
        blank=True,
        null=True
    )
    prediction_time = models.FloatField(
        verbose_name="Waktu Prediksi (detik)",
        blank=True,
        null=True
    )
    total_time = models.FloatField(
        verbose_name="Total Waktu (detik)",
        blank=True,
        null=True
    )
    
    # Additional Info
    notes = models.TextField(
        verbose_name="Catatan",
        blank=True,
        help_text="Catatan tambahan dari sistem atau user"
    )
    is_correct = models.BooleanField(
        verbose_name="Prediksi Benar?",
        default=None,
        null=True,
        blank=True,
        help_text="Feedback dari user/expert"
    )
    actual_disease = models.ForeignKey(
        DiseaseCategory,
        on_delete=models.SET_NULL,
        related_name='actual_diagnoses',
        verbose_name="Penyakit Sebenarnya",
        null=True,
        blank=True,
        help_text="Untuk validasi dan improvement model"
    )
    
    diagnosed_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Waktu Diagnosis"
    )
    
    class Meta:
        verbose_name = "Hasil Diagnosis"
        verbose_name_plural = "Hasil Diagnosis"
        ordering = ['-diagnosed_at']
    
    def __str__(self):
        return f"{self.predicted_disease} - {self.max_confidence:.2f}% - {self.diagnosed_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_all_confidences(self):
        """Return dictionary semua confidence scores"""
        return {
            'Bacterial Leaf Blight': self.confidence_bacterial_blight,
            'Rice Blast': self.confidence_rice_blast,
            'Tungro': self.confidence_tungro,
            'Healthy': self.confidence_healthy,
        }
    
    def is_high_confidence(self):
        """Cek apakah confidence tinggi (>80%)"""
        return self.max_confidence >= 80
    
    def get_confidence_level(self):
        """Return kategori confidence level"""
        if self.max_confidence >= 90:
            return "Sangat Tinggi"
        elif self.max_confidence >= 75:
            return "Tinggi"
        elif self.max_confidence >= 60:
            return "Sedang"
        else:
            return "Rendah"


# ========== MODEL 5: MODEL TRAINING HISTORY ==========
class ModelTrainingHistory(models.Model):
    """
    Model untuk menyimpan riwayat training model ML
    """
    model_name = models.CharField(
        max_length=100,
        verbose_name="Nama Model"
    )
    version = models.CharField(
        max_length=50,
        verbose_name="Versi"
    )
    
    # Dataset Info
    dataset_name = models.CharField(
        max_length=200,
        verbose_name="Nama Dataset",
        default="Rice Leaf and Crop Disease Detection Dataset"
    )
    total_samples = models.IntegerField(
        verbose_name="Total Sampel"
    )
    training_samples = models.IntegerField(
        verbose_name="Sampel Training"
    )
    validation_samples = models.IntegerField(
        verbose_name="Sampel Validasi"
    )
    test_samples = models.IntegerField(
        verbose_name="Sampel Testing"
    )
    
    # SMOTE Applied
    smote_applied = models.BooleanField(
        verbose_name="SMOTE Diterapkan?",
        default=True
    )
    samples_after_smote = models.IntegerField(
        verbose_name="Sampel Setelah SMOTE",
        blank=True,
        null=True
    )
    
    # Model Performance
    accuracy = models.FloatField(
        verbose_name="Akurasi (%)"
    )
    precision = models.FloatField(
        verbose_name="Precision (%)"
    )
    recall = models.FloatField(
        verbose_name="Recall (%)"
    )
    f1_score = models.FloatField(
        verbose_name="F1-Score (%)"
    )
    val_accuracy = models.FloatField(
        verbose_name="Akurasi Validasi (%)",
        blank=True,
        null=True
    )
    per_class_metrics = models.JSONField(
        verbose_name="Metrik per Kelas",
        blank=True,
        null=True,
        help_text="JSON: {class_name: {precision, recall, f1, support}}"
    )
    
    # Random Forest Parameters
    n_estimators = models.IntegerField(
        verbose_name="Jumlah Trees",
        default=100
    )
    max_depth = models.IntegerField(
        verbose_name="Max Depth",
        blank=True,
        null=True
    )
    min_samples_split = models.IntegerField(
        verbose_name="Min Samples Split",
        default=2
    )
    
    # Training Info
    training_duration = models.FloatField(
        verbose_name="Durasi Training (menit)",
        blank=True,
        null=True
    )
    model_file_path = models.CharField(
        max_length=500,
        verbose_name="Path File Model",
        blank=True
    )
    
    notes = models.TextField(
        verbose_name="Catatan",
        blank=True
    )
    is_active = models.BooleanField(
        verbose_name="Model Aktif?",
        default=False,
        help_text="Model yang sedang digunakan untuk prediksi"
    )
    
    trained_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Tanggal Training"
    )
    trained_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        verbose_name="Di-training oleh"
    )
    
    class Meta:
        verbose_name = "Riwayat Training Model"
        verbose_name_plural = "Riwayat Training Model"
        ordering = ['-trained_at']
    
    def __str__(self):
        return f"{self.model_name} {self.version} - Acc: {self.accuracy:.2f}%"
    
    def save(self, *args, **kwargs):
        """Override save untuk set model lain menjadi non-aktif jika ini aktif"""
        if self.is_active:
            ModelTrainingHistory.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)
    
    def get_per_class_list(self):
        """Return per_class_metrics sebagai list siap pakai di template"""
        if not self.per_class_metrics:
            return []
        color_map = {
            'Bacterial Blight': 'danger',
            'Rice Blast':       'warning',
            'Tungro':           'info',
            'Healthy':          'success',
        }
        return [
            {
                'name':      class_name,
                'precision': metrics['precision'],
                'recall':    metrics['recall'],
                'f1':        metrics['f1'],
                'support':   metrics['support'],
                'color':     color_map.get(class_name, 'secondary'),
            }
            for class_name, metrics in self.per_class_metrics.items()
        ]


# ========== MODEL 6: SYSTEM STATISTICS ==========
class SystemStatistics(models.Model):
    """
    Model untuk menyimpan statistik sistem (singleton)
    """
    total_images_uploaded = models.IntegerField(
        verbose_name="Total Gambar Diupload",
        default=0
    )
    total_diagnoses = models.IntegerField(
        verbose_name="Total Diagnosis",
        default=0
    )
    total_bacterial_blight = models.IntegerField(
        verbose_name="Total Bacterial Blight",
        default=0
    )
    total_rice_blast = models.IntegerField(
        verbose_name="Total Rice Blast",
        default=0
    )
    total_tungro = models.IntegerField(
        verbose_name="Total Tungro",
        default=0
    )
    total_healthy = models.IntegerField(
        verbose_name="Total Healthy",
        default=0
    )
    
    average_confidence = models.FloatField(
        verbose_name="Rata-rata Confidence (%)",
        default=0
    )
    average_processing_time = models.FloatField(
        verbose_name="Rata-rata Waktu Proses (detik)",
        default=0
    )
    
    last_updated = models.DateTimeField(
        auto_now=True,
        verbose_name="Terakhir Diupdate"
    )
    
    class Meta:
        verbose_name = "Statistik Sistem"
        verbose_name_plural = "Statistik Sistem"
    
    def __str__(self):
        return f"Statistics - {self.total_diagnoses} diagnoses"
    
    @classmethod
    def get_stats(cls):
        """Get or create singleton instance"""
        stats, created = cls.objects.get_or_create(pk=1)
        return stats
    
    def update_statistics(self):
        """Update statistik dari database"""
        from django.db.models import Avg, Count
        
        self.total_images_uploaded = RiceLeafImage.objects.count()
        self.total_diagnoses = DiagnosisResult.objects.count()
        
        # Count per disease
        disease_counts = DiagnosisResult.objects.values(
            'predicted_disease__name'
        ).annotate(count=Count('id'))
        
        for item in disease_counts:
            disease_name = item['predicted_disease__name']
            count = item['count']
            
            if disease_name == 'bacterial_blight':
                self.total_bacterial_blight = count
            elif disease_name == 'rice_blast':
                self.total_rice_blast = count
            elif disease_name == 'tungro':
                self.total_tungro = count
            elif disease_name == 'healthy':
                self.total_healthy = count
        
        # Average metrics
        avg_data = DiagnosisResult.objects.aggregate(
            avg_confidence=Avg('max_confidence'),
            avg_time=Avg('total_time')
        )
        
        self.average_confidence = avg_data['avg_confidence'] or 0
        self.average_processing_time = avg_data['avg_time'] or 0
        
        self.save()