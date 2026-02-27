from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Count, Avg
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    DiseaseCategory,
    RiceLeafImage,
    GLCMFeatures,
    DiagnosisResult,
    ModelTrainingHistory,
    SystemStatistics
)


# ========== DISEASE CATEGORY ADMIN ==========
@admin.register(DiseaseCategory)
class DiseaseCategoryAdmin(admin.ModelAdmin):
    list_display = [
        'display_name',
        'name',
        'severity_badge',
        'total_diagnoses',
        'show_reference_image',
        'updated_at'
    ]
    list_filter = ['severity_level', 'created_at']
    search_fields = ['name', 'display_name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'show_large_image']
    
    fieldsets = (
        ('Informasi Dasar', {
            'fields': ('name', 'display_name', 'severity_level')
        }),
        ('Deskripsi Penyakit', {
            'fields': ('description', 'symptoms', 'causes', 'treatment')
        }),
        ('Gambar Referensi', {
            'fields': ('reference_image', 'show_large_image')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def severity_badge(self, obj):
        """Display severity level dengan badge warna"""
        colors = {
            0: '#6c757d',  # Gray
            1: '#28a745',  # Green
            2: '#ffc107',  # Yellow
            3: '#dc3545',  # Red
        }
        labels = {
            0: 'N/A',
            1: 'Ringan',
            2: 'Sedang',
            3: 'Berat',
        }
        color = colors.get(obj.severity_level, '#6c757d')
        label = labels.get(obj.severity_level, 'N/A')
        
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color, label
        )
    severity_badge.short_description = 'Tingkat Keparahan'
    
    def total_diagnoses(self, obj):
        """Tampilkan jumlah diagnosis"""
        count = obj.get_sample_count()
        return format_html(
            '<strong style="color: #007bff;">{}</strong> diagnoses',
            count
        )
    total_diagnoses.short_description = 'Total Diagnosis'
    
    def show_reference_image(self, obj):
        """Tampilkan thumbnail gambar referensi"""
        if obj.reference_image:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; '
                'object-fit: cover; border-radius: 5px;" />',
                obj.reference_image.url
            )
        return format_html('<span style="color: #999;">No Image</span>')
    show_reference_image.short_description = 'Gambar'
    
    def show_large_image(self, obj):
        """Tampilkan gambar besar di detail page"""
        if obj.reference_image:
            return format_html(
                '<img src="{}" style="max-width: 500px; max-height: 400px; '
                'border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />',
                obj.reference_image.url
            )
        return "Belum ada gambar"
    show_large_image.short_description = 'Preview Gambar'


# ========== RICE LEAF IMAGE ADMIN ==========
@admin.register(RiceLeafImage)
class RiceLeafImageAdmin(admin.ModelAdmin):
    list_display = [
        'image_thumbnail',
        'original_filename',
        'show_dimensions',
        'file_size_display',
        'user',
        'uploaded_at',
        'has_diagnosis'
    ]
    list_filter = ['uploaded_at', 'user']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['uploaded_at', 'file_size', 'image_width', 
                       'image_height', 'show_full_image', 'ip_address']
    date_hierarchy = 'uploaded_at'
    
    fieldsets = (
        ('Gambar', {
            'fields': ('image', 'show_full_image')
        }),
        ('Informasi File', {
            'fields': ('original_filename', 'file_size', 'image_width', 'image_height')
        }),
        ('Informasi Upload', {
            'fields': ('user', 'uploaded_at', 'ip_address')
        }),
    )
    
    def image_thumbnail(self, obj):
        """Tampilkan thumbnail"""
        if obj.image:
            return format_html(
                '<img src="{}" style="width: 60px; height: 60px; '
                'object-fit: cover; border-radius: 5px; '
                'box-shadow: 0 1px 3px rgba(0,0,0,0.2);" />',
                obj.image.url
            )
        return "No Image"
    image_thumbnail.short_description = 'Preview'
    
    def show_full_image(self, obj):
        """Tampilkan gambar penuh"""
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width: 600px; border-radius: 8px; '
                'box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />',
                obj.image.url
            )
        return "No Image"
    show_full_image.short_description = 'Gambar Lengkap'
    
    def show_dimensions(self, obj):
        """Tampilkan dimensi gambar"""
        if obj.image_width and obj.image_height:
            return f"{obj.image_width} × {obj.image_height} px"
        return "-"
    show_dimensions.short_description = 'Dimensi'
    
    def file_size_display(self, obj):
        """Tampilkan ukuran file dalam format readable"""
        size_mb = obj.get_file_size_mb()
        if size_mb >= 1:
            return f"{size_mb} MB"
        elif obj.file_size:
            return f"{round(obj.file_size / 1024, 2)} KB"
        return "-"
    file_size_display.short_description = 'Ukuran File'
    
    def has_diagnosis(self, obj):
        """Cek apakah sudah ada diagnosis"""
        count = obj.diagnoses.count()
        if count > 0:
            return format_html(
                '<span style="color: green;">✓ {} diagnosis</span>',
                count
            )
        return format_html('<span style="color: orange;">✗ Belum</span>')
    has_diagnosis.short_description = 'Status Diagnosis'


# ========== GLCM FEATURES ADMIN ==========
@admin.register(GLCMFeatures)
class GLCMFeaturesAdmin(admin.ModelAdmin):
    list_display = [
        'image_filename',
        'show_contrast',
        'show_energy',
        'show_homogeneity',
        'extraction_time',
        'created_at'
    ]
    list_filter = ['created_at']
    search_fields = ['image__original_filename']
    readonly_fields = [
        'image', 'created_at',
        'contrast_0', 'dissimilarity_0', 'homogeneity_0', 
        'energy_0', 'correlation_0', 'asm_0',
        'contrast_45', 'dissimilarity_45', 'homogeneity_45',
        'energy_45', 'correlation_45', 'asm_45',
        'contrast_90', 'dissimilarity_90', 'homogeneity_90',
        'energy_90', 'correlation_90', 'asm_90',
        'contrast_135', 'dissimilarity_135', 'homogeneity_135',
        'energy_135', 'correlation_135', 'asm_135',
        'contrast_mean', 'dissimilarity_mean', 'homogeneity_mean',
        'energy_mean', 'correlation_mean', 'asm_mean',
        'extraction_time'
    ]
    
    fieldsets = (
        ('Informasi Gambar', {
            'fields': ('image', 'extraction_time', 'created_at')
        }),
        ('GLCM Features - 0°', {
            'fields': ('contrast_0', 'dissimilarity_0', 'homogeneity_0',
                      'energy_0', 'correlation_0', 'asm_0'),
            'classes': ('collapse',)
        }),
        ('GLCM Features - 45°', {
            'fields': ('contrast_45', 'dissimilarity_45', 'homogeneity_45',
                      'energy_45', 'correlation_45', 'asm_45'),
            'classes': ('collapse',)
        }),
        ('GLCM Features - 90°', {
            'fields': ('contrast_90', 'dissimilarity_90', 'homogeneity_90',
                      'energy_90', 'correlation_90', 'asm_90'),
            'classes': ('collapse',)
        }),
        ('GLCM Features - 135°', {
            'fields': ('contrast_135', 'dissimilarity_135', 'homogeneity_135',
                      'energy_135', 'correlation_135', 'asm_135'),
            'classes': ('collapse',)
        }),
        ('Average Features', {
            'fields': ('contrast_mean', 'dissimilarity_mean', 'homogeneity_mean',
                      'energy_mean', 'correlation_mean', 'asm_mean')
        }),
    )
    
    def image_filename(self, obj):
        return obj.image.original_filename
    image_filename.short_description = 'Nama File'
    
    def show_contrast(self, obj):
        return f"{obj.contrast_mean:.4f}"
    show_contrast.short_description = 'Contrast (Avg)'
    
    def show_energy(self, obj):
        return f"{obj.energy_mean:.4f}"
    show_energy.short_description = 'Energy (Avg)'
    
    def show_homogeneity(self, obj):
        return f"{obj.homogeneity_mean:.4f}"
    show_homogeneity.short_description = 'Homogeneity (Avg)'


# ========== DIAGNOSIS RESULT ADMIN ==========
@admin.register(DiagnosisResult)
class DiagnosisResultAdmin(admin.ModelAdmin):
    list_display = [
        'image_thumb',
        'predicted_disease',
        'confidence_display',
        'confidence_level_badge',
        'processing_time',
        'feedback_status',
        'diagnosed_at'
    ]
    list_filter = [
        'predicted_disease',
        'diagnosed_at',
        'is_correct',
        'model_version'
    ]
    search_fields = [
        'image__original_filename',
        'predicted_disease__display_name',
        'notes'
    ]
    readonly_fields = [
        'image', 'predicted_disease', 'max_confidence',
        'confidence_bacterial_blight', 'confidence_rice_blast',
        'confidence_tungro', 'confidence_healthy',
        'model_name', 'model_version',
        'preprocessing_time', 'prediction_time', 'total_time',
        'diagnosed_at', 'show_confidence_chart'
    ]
    date_hierarchy = 'diagnosed_at'
    
    fieldsets = (
        ('Hasil Prediksi', {
            'fields': ('image', 'predicted_disease', 'max_confidence',
                      'show_confidence_chart')
        }),
        ('Confidence Scores Per Kelas', {
            'fields': ('confidence_bacterial_blight', 'confidence_rice_blast',
                      'confidence_tungro', 'confidence_healthy'),
            'classes': ('collapse',)
        }),
        ('Informasi Model', {
            'fields': ('model_name', 'model_version')
        }),
        ('Waktu Pemrosesan', {
            'fields': ('preprocessing_time', 'prediction_time', 'total_time'),
            'classes': ('collapse',)
        }),
        ('Validasi & Feedback', {
            'fields': ('is_correct', 'actual_disease', 'notes')
        }),
        ('Metadata', {
            'fields': ('diagnosed_at',),
            'classes': ('collapse',)
        }),
    )
    
    def image_thumb(self, obj):
        """Thumbnail gambar"""
        if obj.image and obj.image.image:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; '
                'object-fit: cover; border-radius: 5px;" />',
                obj.image.image.url
            )
        return "No Image"
    image_thumb.short_description = 'Gambar'
    
    def confidence_display(self, obj):
        """Display confidence dengan progress bar"""
        return format_html(
            '<div style="width: 100px; background-color: #e9ecef; '
            'border-radius: 3px; overflow: hidden;">'
            '<div style="width: {}%; background-color: {}; color: white; '
            'text-align: center; padding: 2px 0; font-size: 11px; font-weight: bold;">'
            '{:.1f}%</div></div>',
            obj.max_confidence,
            '#28a745' if obj.max_confidence >= 80 else '#ffc107' if obj.max_confidence >= 60 else '#dc3545',
            obj.max_confidence
        )
    confidence_display.short_description = 'Confidence'
    
    def confidence_level_badge(self, obj):
        """Badge untuk confidence level"""
        level = obj.get_confidence_level()
        colors = {
            'Sangat Tinggi': '#28a745',
            'Tinggi': '#17a2b8',
            'Sedang': '#ffc107',
            'Rendah': '#dc3545'
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px; font-size: 11px;">{}</span>',
            colors.get(level, '#6c757d'),
            level
        )
    confidence_level_badge.short_description = 'Level'
    
    def processing_time(self, obj):
        """Display total processing time"""
        if obj.total_time:
            return f"{obj.total_time:.2f}s"
        return "-"
    processing_time.short_description = 'Waktu Proses'
    
    def feedback_status(self, obj):
        """Status feedback dari user"""
        if obj.is_correct is None:
            return format_html('<span style="color: #6c757d;">Belum divalidasi</span>')
        elif obj.is_correct:
            return format_html('<span style="color: green;">✓ Benar</span>')
        else:
            return format_html('<span style="color: red;">✗ Salah</span>')
    feedback_status.short_description = 'Feedback'
    
    def show_confidence_chart(self, obj):
        """Tampilkan bar chart confidence scores"""
        confidences = obj.get_all_confidences()
        
        html = '<div style="max-width: 500px;">'
        for disease, conf in confidences.items():
            color = '#007bff' if conf == obj.max_confidence else '#e9ecef'
            text_color = 'white' if conf == obj.max_confidence else '#333'
            
            html += f'''
            <div style="margin-bottom: 8px;">
                <div style="font-size: 12px; margin-bottom: 2px;">{disease}</div>
                <div style="width: 100%; background-color: #f8f9fa; border-radius: 3px; overflow: hidden;">
                    <div style="width: {conf}%; background-color: {color}; 
                         color: {text_color}; padding: 4px 8px; font-size: 11px; font-weight: bold;">
                        {conf:.2f}%
                    </div>
                </div>
            </div>
            '''
        
        html += '</div>'
        return mark_safe(html)
    show_confidence_chart.short_description = 'Confidence Chart'


# ========== MODEL TRAINING HISTORY ADMIN ==========
@admin.register(ModelTrainingHistory)
class ModelTrainingHistoryAdmin(admin.ModelAdmin):
    list_display = [
        'model_version_display',
        'accuracy_display',
        'f1_display',
        'total_samples',
        'smote_status',
        'active_badge',
        'trained_at'
    ]
    list_filter = ['is_active', 'smote_applied', 'trained_at']
    search_fields = ['model_name', 'version', 'notes']
    readonly_fields = [
        'trained_at', 'trained_by',
        'show_performance_metrics', 'show_dataset_info'
    ]
    date_hierarchy = 'trained_at'
    
    fieldsets = (
        ('Informasi Model', {
            'fields': ('model_name', 'version', 'is_active', 'trained_by', 'trained_at')
        }),
        ('Dataset Information', {
            'fields': ('show_dataset_info', 'dataset_name', 'total_samples',
                      'training_samples', 'validation_samples', 'test_samples')
        }),
        ('SMOTE Configuration', {
            'fields': ('smote_applied', 'samples_after_smote')
        }),
        ('Performance Metrics', {
            'fields': ('show_performance_metrics', 'accuracy', 'precision',
                      'recall', 'f1_score')
        }),
        ('Random Forest Parameters', {
            'fields': ('n_estimators', 'max_depth', 'min_samples_split'),
            'classes': ('collapse',)
        }),
        ('Training Details', {
            'fields': ('training_duration', 'model_file_path', 'notes'),
            'classes': ('collapse',)
        }),
    )
    
    def model_version_display(self, obj):
        return f"{obj.model_name} {obj.version}"
    model_version_display.short_description = 'Model'
    
    def accuracy_display(self, obj):
        """Display accuracy dengan warna"""
        color = '#28a745' if obj.accuracy >= 90 else '#ffc107' if obj.accuracy >= 80 else '#dc3545'
        return format_html(
            '<strong style="color: {};">{:.2f}%</strong>',
            color, obj.accuracy
        )
    accuracy_display.short_description = 'Accuracy'
    
    def f1_display(self, obj):
        """Display F1-Score"""
        return f"{obj.f1_score:.2f}%"
    f1_display.short_description = 'F1-Score'
    
    def smote_status(self, obj):
        """Status SMOTE"""
        if obj.smote_applied:
            return format_html(
                '<span style="color: green;">✓ Applied</span>'
            )
        return format_html('<span style="color: #999;">✗ Not Applied</span>')
    smote_status.short_description = 'SMOTE'
    
    def active_badge(self, obj):
        """Badge untuk model aktif"""
        if obj.is_active:
            return format_html(
                '<span style="background-color: #28a745; color: white; '
                'padding: 3px 10px; border-radius: 3px; font-weight: bold;">ACTIVE</span>'
            )
        return format_html(
            '<span style="color: #999;">Inactive</span>'
        )
    active_badge.short_description = 'Status'
    
    def show_performance_metrics(self, obj):
        """Tampilkan metrics dalam format tabel"""
        return format_html('''
            <table style="border-collapse: collapse; width: 100%; max-width: 400px;">
                <tr>
                    <th style="padding: 8px; background-color: #f8f9fa; border: 1px solid #dee2e6;">Metric</th>
                    <th style="padding: 8px; background-color: #f8f9fa; border: 1px solid #dee2e6;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">Accuracy</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>{:.2f}%</strong></td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">Precision</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">{:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">Recall</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">{:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">F1-Score</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>{:.2f}%</strong></td>
                </tr>
            </table>
        ''', obj.accuracy, obj.precision, obj.recall, obj.f1_score)
    show_performance_metrics.short_description = 'Performance Metrics'
    
    def show_dataset_info(self, obj):
        """Tampilkan info dataset"""
        return format_html('''
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; max-width: 500px;">
                <p><strong>Dataset:</strong> {}</p>
                <p><strong>Total Samples:</strong> {:,}</p>
                <p><strong>Training:</strong> {:,} samples</p>
                <p><strong>Validation:</strong> {:,} samples</p>
                <p><strong>Testing:</strong> {:,} samples</p>
                {}
            </div>
        ''',
            obj.dataset_name,
            obj.total_samples,
            obj.training_samples,
            obj.validation_samples,
            obj.test_samples,
            f'<p><strong>After SMOTE:</strong> {obj.samples_after_smote:,} samples</p>' if obj.smote_applied else ''
        )
    show_dataset_info.short_description = 'Dataset Information'


# ========== SYSTEM STATISTICS ADMIN ==========
@admin.register(SystemStatistics)
class SystemStatisticsAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'total_diagnoses',
        'average_confidence',
        'average_processing_time',
        'last_updated'
    ]
    readonly_fields = [
        'total_images_uploaded',
        'total_diagnoses',
        'total_bacterial_blight',
        'total_rice_blast',
        'total_tungro',
        'total_healthy',
        'average_confidence',
        'average_processing_time',
        'last_updated',
        'show_disease_distribution',
        'show_statistics_summary'
    ]
    
    fieldsets = (
        ('Overview', {
            'fields': ('show_statistics_summary',)
        }),
        ('Total Counts', {
            'fields': ('total_images_uploaded', 'total_diagnoses')
        }),
        ('Disease Distribution', {
            'fields': ('show_disease_distribution', 'total_bacterial_blight',
                      'total_rice_blast', 'total_tungro', 'total_healthy', 'total_rice')
        }),
        ('Average Metrics', {
            'fields': ('average_confidence', 'average_processing_time')
        }),
        ('Metadata', {
            'fields': ('last_updated',)
        }),
    )
    
    def has_add_permission(self, request):
        """Prevent creating multiple instances"""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """Prevent deleting the singleton"""
        return False
    
    def show_disease_distribution(self, obj):
        """Tampilkan distribusi penyakit dalam chart"""
        diseases = [
            ('Bacterial Blight', obj.total_bacterial_blight, '#dc3545'),
            ('Rice Blast', obj.total_rice_blast, '#ffc107'),
            ('Tungro', obj.total_tungro, '#17a2b8'),
            ('Healthy', obj.total_healthy, '#28a745'),
            ('Rice', obj.total_rice, '#6c757d'),
        ]
        
        total = obj.total_diagnoses
        if total == 0:
            return "Belum ada data diagnosis"
        
        html = '<div style="max-width: 600px;">'
        for name, count, color in diseases:
            percentage = (count / total * 100) if total > 0 else 0
            html += f'''
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
                    <span><strong>{name}</strong></span>
                    <span>{count} ({percentage:.1f}%)</span>
                </div>
                <div style="width: 100%; background-color: #e9ecef; border-radius: 3px; overflow: hidden;">
                    <div style="width: {percentage}%; background-color: {color}; height: 20px;"></div>
                </div>
            </div>
            '''
        html += '</div>'
        return mark_safe(html)
    show_disease_distribution.short_description = 'Disease Distribution'
    
    def show_statistics_summary(self, obj):
        """Tampilkan summary statistik dalam cards"""
        return format_html('''
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     padding: 20px; border-radius: 8px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; opacity: 0.9;">Total Images</div>
                    <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{:,}</div>
                </div>
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                     padding: 20px; border-radius: 8px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; opacity: 0.9;">Total Diagnoses</div>
                    <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{:,}</div>
                </div>
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                     padding: 20px; border-radius: 8px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; opacity: 0.9;">Avg Confidence</div>
                    <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{:.1f}%</div>
                </div>
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                     padding: 20px; border-radius: 8px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; opacity: 0.9;">Avg Process Time</div>
                    <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{:.2f}s</div>
                </div>
            </div>
        ''',
            obj.total_images_uploaded,
            obj.total_diagnoses,
            obj.average_confidence,
            obj.average_processing_time
        )
    show_statistics_summary.short_description = 'Statistics Overview'


# ========== CUSTOM ADMIN ACTIONS ==========
@admin.action(description='Update System Statistics')
def update_statistics(modeladmin, request, queryset):
    """Action untuk update statistik sistem"""
    stats = SystemStatistics.get_stats()
    stats.update_statistics()
    modeladmin.message_user(request, "System statistics updated successfully!")

# Tambahkan action ke admin
SystemStatisticsAdmin.actions = [update_statistics]