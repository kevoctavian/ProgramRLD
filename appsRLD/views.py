"""
Views untuk Aplikasi Klasifikasi Penyakit Daun Padi
Menggunakan Class-Based Views (CBV)
"""

from multiprocessing import context

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views import View
from django.views.generic import TemplateView, ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Avg, Q
from django.core.paginator import Paginator
from django.utils import timezone
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import base64
import joblib
from .models import (
    RiceLeafImage,
    DiagnosisResult,
    DiseaseCategory,
    GLCMFeatures,
    SystemStatistics
)
from .forms import ImageUploadForm, FeedbackForm, SearchForm, RegisterForm, LoginForm
from .ml_pipeline import RiceDiseasePipeline
from datetime import timedelta
from django.db.models.functions import TruncMonth
from PIL import Image
import os
import json


# Initialize ML Pipeline
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models')
pipeline = RiceDiseasePipeline()

try:
    if os.path.exists(MODEL_PATH):
        pipeline.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Model not found. Please train the model first.")
except Exception as e:
    print(f"⚠ Error loading model: {e}")

# ========== HOME PAGE ==========
class HomeView(TemplateView):
    """
    Homepage dengan overview dan quick stats
    """
    template_name = 'appsRLD/home.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        stats = SystemStatistics.get_stats()
        stats.update_statistics()

        # Get model accuracy from latest training
        try:
            from .models import ModelTrainingHistory
            latest_training = ModelTrainingHistory.objects.filter(
                is_active=True
            ).order_by('-trained_at').first()
            
            model_accuracy = latest_training.accuracy if latest_training else 0
        except Exception as e:
            print(f"Error getting model accuracy: {e}")
            model_accuracy = 0

        recent_diagnoses = DiagnosisResult.objects.select_related(
            'image', 'predicted_disease'
        ).order_by('-diagnosed_at')[:6]

        disease_distribution = DiagnosisResult.objects.filter(
            predicted_disease__isnull=False
        ).values(
            'predicted_disease__display_name'
        ).annotate(count=Count('id')).order_by('-count')

        context.update({
            'stats': stats,
            'model_accuracy': model_accuracy,
            'recent_diagnoses': recent_diagnoses,
            'disease_distribution': list(disease_distribution),
            'model_loaded': pipeline.model is not None
        })
        return context


# ========== UPLOAD & DIAGNOSIS ==========
class UploadAndDiagnoseView(View):
    """
    Upload gambar dan lakukan diagnosis
    """
    template_name = 'appsRLD/upload.html'
    login_url = '/login/'

    def get(self, request):
        form = ImageUploadForm()
        return render(request, self.template_name, {
            'form': form,
            'model_loaded': pipeline.model is not None
        })

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            if pipeline.model is None:
                messages.error(
                    request,
                    "Model ML belum di-load. Silakan hubungi administrator."
                )
                return redirect('appsRLD:upload')

            try:
                rice_image = form.save(commit=False)

                if request.user.is_authenticated:
                    rice_image.user = request.user

                x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                if x_forwarded_for:
                    rice_image.ip_address = x_forwarded_for.split(',')[0]
                else:
                    rice_image.ip_address = request.META.get('REMOTE_ADDR')

                rice_image.original_filename = request.FILES['image'].name
                rice_image.file_size = request.FILES['image'].size

                img = Image.open(request.FILES['image'])
                rice_image.image_width, rice_image.image_height = img.size
                rice_image.save()

                result = pipeline.predict_single_image(img)

                GLCMFeatures.objects.create(
                    image=rice_image,
                    contrast_0=result['glcm_features']['contrast_0'],
                    dissimilarity_0=result['glcm_features']['dissimilarity_0'],
                    homogeneity_0=result['glcm_features']['homogeneity_0'],
                    energy_0=result['glcm_features']['energy_0'],
                    correlation_0=result['glcm_features']['correlation_0'],
                    asm_0=result['glcm_features']['ASM_0'],
                    contrast_45=result['glcm_features']['contrast_45'],
                    dissimilarity_45=result['glcm_features']['dissimilarity_45'],
                    homogeneity_45=result['glcm_features']['homogeneity_45'],
                    energy_45=result['glcm_features']['energy_45'],
                    correlation_45=result['glcm_features']['correlation_45'],
                    asm_45=result['glcm_features']['ASM_45'],
                    contrast_90=result['glcm_features']['contrast_90'],
                    dissimilarity_90=result['glcm_features']['dissimilarity_90'],
                    homogeneity_90=result['glcm_features']['homogeneity_90'],
                    energy_90=result['glcm_features']['energy_90'],
                    correlation_90=result['glcm_features']['correlation_90'],
                    asm_90=result['glcm_features']['ASM_90'],
                    contrast_135=result['glcm_features']['contrast_135'],
                    dissimilarity_135=result['glcm_features']['dissimilarity_135'],
                    homogeneity_135=result['glcm_features']['homogeneity_135'],
                    energy_135=result['glcm_features']['energy_135'],
                    correlation_135=result['glcm_features']['correlation_135'],
                    asm_135=result['glcm_features']['ASM_135'],
                    contrast_mean=result['glcm_features']['contrast_mean'],
                    dissimilarity_mean=result['glcm_features']['dissimilarity_mean'],
                    homogeneity_mean=result['glcm_features']['homogeneity_mean'],
                    energy_mean=result['glcm_features']['energy_mean'],
                    correlation_mean=result['glcm_features']['correlation_mean'],
                    asm_mean=result['glcm_features']['ASM_mean'],
                    extraction_time=result['glcm_features']['extraction_time']
                )

                predicted_disease = DiseaseCategory.objects.get(
                    name=result['predicted_class']
                )

                diagnosis = DiagnosisResult.objects.create(
                    image=rice_image,
                    predicted_disease=predicted_disease,
                    confidence_bacterial_blight=result['all_probabilities']['bacterial_blight'],
                    confidence_rice_blast=result['all_probabilities']['rice_blast'],
                    confidence_tungro=result['all_probabilities']['tungro'],
                    confidence_healthy=result['all_probabilities']['healthy'],
                    max_confidence=result['confidence'],
                    preprocessing_time=result['preprocessing_time'],
                    prediction_time=result['prediction_time'],
                    total_time=result['total_time']
                )

                stats = SystemStatistics.get_stats()
                stats.update_statistics()

                messages.success(
                    request,
                    f"Diagnosis berhasil! Terdeteksi: {predicted_disease.display_name} "
                    f"dengan confidence {result['confidence']:.2f}%"
                )

                return redirect('appsRLD:result', diagnosis_id=diagnosis.id)

            except Exception as e:
                messages.error(request, f"Error saat melakukan diagnosis: {str(e)}")
                return redirect('appsRLD:upload')

        return render(request, self.template_name, {
            'form': form,
            'model_loaded': pipeline.model is not None
        })

# ========== CAMERA CAPTURE ==========
class CameraCaptureDiagnoseView(LoginRequiredMixin, View):
    """
    Terima foto dari kamera (base64) dan lakukan diagnosis
    """
    login_url = '/login/'

    def post(self, request):
        try:
            import base64
            from django.core.files.base import ContentFile

            # Ambil data base64 dari request
            image_data = request.POST.get('image_data', '')

            if not image_data:
                messages.error(request, "Tidak ada gambar yang diterima dari kamera.")
                return redirect('appsRLD:upload')

            # Decode base64 → file
            # Format: "data:image/png;base64,xxxx..."
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]

            image_bytes = base64.b64decode(image_data)
            image_file = ContentFile(image_bytes, name='camera_capture.png')

            if pipeline.model is None:
                messages.error(request, "Model ML belum di-load.")
                return redirect('appsRLD:upload')

            # Simpan gambar
            rice_image = RiceLeafImage()
            rice_image.image.save('camera_capture.png', image_file, save=False)

            if request.user.is_authenticated:
                rice_image.user = request.user

            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            rice_image.ip_address = (
                x_forwarded_for.split(',')[0]
                if x_forwarded_for
                else request.META.get('REMOTE_ADDR')
            )

            rice_image.original_filename = 'camera_capture.png'
            rice_image.file_size = len(image_bytes)

            img = Image.open(image_file)
            rice_image.image_width, rice_image.image_height = img.size
            rice_image.save()

            # Prediksi
            image_file.seek(0)
            img = Image.open(image_file)
            result = pipeline.predict_single_image(img)

            # Simpan GLCM
            GLCMFeatures.objects.create(
                image=rice_image,
                contrast_0=result['glcm_features']['contrast_0'],
                dissimilarity_0=result['glcm_features']['dissimilarity_0'],
                homogeneity_0=result['glcm_features']['homogeneity_0'],
                energy_0=result['glcm_features']['energy_0'],
                correlation_0=result['glcm_features']['correlation_0'],
                asm_0=result['glcm_features']['ASM_0'],
                contrast_45=result['glcm_features']['contrast_45'],
                dissimilarity_45=result['glcm_features']['dissimilarity_45'],
                homogeneity_45=result['glcm_features']['homogeneity_45'],
                energy_45=result['glcm_features']['energy_45'],
                correlation_45=result['glcm_features']['correlation_45'],
                asm_45=result['glcm_features']['ASM_45'],
                contrast_90=result['glcm_features']['contrast_90'],
                dissimilarity_90=result['glcm_features']['dissimilarity_90'],
                homogeneity_90=result['glcm_features']['homogeneity_90'],
                energy_90=result['glcm_features']['energy_90'],
                correlation_90=result['glcm_features']['correlation_90'],
                asm_90=result['glcm_features']['ASM_90'],
                contrast_135=result['glcm_features']['contrast_135'],
                dissimilarity_135=result['glcm_features']['dissimilarity_135'],
                homogeneity_135=result['glcm_features']['homogeneity_135'],
                energy_135=result['glcm_features']['energy_135'],
                correlation_135=result['glcm_features']['correlation_135'],
                asm_135=result['glcm_features']['ASM_135'],
                contrast_mean=result['glcm_features']['contrast_mean'],
                dissimilarity_mean=result['glcm_features']['dissimilarity_mean'],
                homogeneity_mean=result['glcm_features']['homogeneity_mean'],
                energy_mean=result['glcm_features']['energy_mean'],
                correlation_mean=result['glcm_features']['correlation_mean'],
                asm_mean=result['glcm_features']['ASM_mean'],
                extraction_time=result['glcm_features']['extraction_time']
            )

            predicted_disease = DiseaseCategory.objects.get(
                name=result['predicted_class']
            )

            diagnosis = DiagnosisResult.objects.create(
                image=rice_image,
                predicted_disease=predicted_disease,
                confidence_bacterial_blight=result['all_probabilities']['bacterial_blight'],
                confidence_rice_blast=result['all_probabilities']['rice_blast'],
                confidence_tungro=result['all_probabilities']['tungro'],
                confidence_healthy=result['all_probabilities']['healthy'],
                max_confidence=result['confidence'],
                preprocessing_time=result['preprocessing_time'],
                prediction_time=result['prediction_time'],
                total_time=result['total_time']
            )

            stats = SystemStatistics.get_stats()
            stats.update_statistics()

            messages.success(
                request,
                f"Diagnosis berhasil! Terdeteksi: {predicted_disease.display_name} "
                f"dengan confidence {result['confidence']:.2f}%"
            )

            return redirect('appsRLD:result', diagnosis_id=diagnosis.id)

        except Exception as e:
            messages.error(request, f"Error saat memproses foto kamera: {str(e)}")
            return redirect('appsRLD:upload')

# ========== DIAGNOSIS RESULT ==========
class DiagnosisResultView(View):
    """
    Tampilkan hasil diagnosis
    """
    template_name = 'appsRLD/result.html'
    login_url = '/login/'

    def get(self, request, diagnosis_id):
        diagnosis = get_object_or_404(
            DiagnosisResult.objects.select_related(
                'image', 'predicted_disease', 'actual_disease'
            ),
            id=diagnosis_id
        )

        try:
            glcm_features = diagnosis.image.glcm_features
        except Exception:
            glcm_features = None

        confidence_data = diagnosis.get_all_confidences()
        feedback_form = FeedbackForm(instance=diagnosis)

        return render(request, self.template_name, {
            'diagnosis': diagnosis,
            'glcm_features': glcm_features,
            'confidence_data': json.dumps(confidence_data),
            'feedback_form': feedback_form
        })

    def post(self, request, diagnosis_id):
        diagnosis = get_object_or_404(DiagnosisResult, id=diagnosis_id)
        feedback_form = FeedbackForm(request.POST, instance=diagnosis)

        if feedback_form.is_valid():
            feedback_form.save()
            messages.success(request, "Terima kasih atas feedback Anda!")
            return redirect('appsRLD:result', diagnosis_id=diagnosis.id)

        try:
            glcm_features = diagnosis.image.glcm_features
        except Exception:
            glcm_features = None

        return render(request, self.template_name, {
            'diagnosis': diagnosis,
            'glcm_features': glcm_features,
            'confidence_data': json.dumps(diagnosis.get_all_confidences()),
            'feedback_form': feedback_form
        })


# # ========== HISTORY ==========
class DiagnosisHistoryView(View):
    """
    Tampilkan riwayat diagnosis dengan filter dan search
    """
    template_name = 'appsRLD/history.html'
    login_url = '/login/'

    def get(self, request):
        diagnoses = DiagnosisResult.objects.select_related(
            'image', 'predicted_disease'
        ).order_by('-diagnosed_at')

        search_form = SearchForm(request.GET)

        if search_form.is_valid():
            disease = search_form.cleaned_data.get('disease')
            if disease:
                diagnoses = diagnoses.filter(predicted_disease=disease)

            date_from = search_form.cleaned_data.get('date_from')
            if date_from:
                diagnoses = diagnoses.filter(diagnosed_at__date__gte=date_from)

            date_to = search_form.cleaned_data.get('date_to')
            if date_to:
                diagnoses = diagnoses.filter(diagnosed_at__date__lte=date_to)

            min_confidence = search_form.cleaned_data.get('min_confidence')
            if min_confidence:
                diagnoses = diagnoses.filter(max_confidence__gte=min_confidence)

            search_query = search_form.cleaned_data.get('search_query')
            if search_query:
                diagnoses = diagnoses.filter(
                    image__original_filename__icontains=search_query
                )

        paginator = Paginator(diagnoses, 12)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'search_form': search_form,
            'total_count': diagnoses.count()
        })

# ========== DELETE DIAGNOSIS ==========
class DeleteDiagnosisView(View):
    """
    Hapus diagnosis (untuk authenticated user)
    """
    def get(self, request, diagnosis_id):
        diagnosis = get_object_or_404(DiagnosisResult, id=diagnosis_id)

        # Allow delete for: admin, staff, or image owner
        if request.user.is_staff or (
            request.user.is_authenticated and 
            diagnosis.image.user and 
            diagnosis.image.user == request.user
        ):
            image = diagnosis.image
            diagnosis.delete()
            
            # Delete image if no more diagnoses reference it
            if image.diagnoses.count() == 0:
                image.delete()
            
            messages.success(request, "✓ Diagnosis berhasil dihapus.")
        else:
            messages.error(request, "✗ Anda tidak memiliki izin untuk menghapus diagnosis ini.")

        return redirect('appsRLD:history')

# ========== CLEAR ALL HISTORY ==========
class ClearAllHistoryView(View):
    """
    Hapus semua history diagnosis (admin only atau confirm)
    """
    def post(self, request):
        # Check if user confirmed
        confirm = request.POST.get('confirm', '')
        
        if confirm != 'HAPUS':
            messages.error(request, "Konfirmasi gagal. Ketik 'HAPUS' untuk menghapus semua data.")
            return redirect('appsRLD:history')
        
        # Delete all diagnoses and images
        try:
            diagnosis_count = DiagnosisResult.objects.count()
            image_count = RiceLeafImage.objects.count()
            
            DiagnosisResult.objects.all().delete()
            RiceLeafImage.objects.all().delete()
            
            # Update statistics
            stats = SystemStatistics.get_stats()
            stats.update_statistics()
            
            messages.success(
                request, 
                f"✓ Berhasil menghapus {diagnosis_count} diagnosis dan {image_count} images."
            )
        except Exception as e:
            messages.error(request, f"✗ Error: {str(e)}")
        
        return redirect('appsRLD:history')

# ========== DISEASE INFO ==========
class DiseaseListView(ListView):
    """
    Tampilkan daftar semua penyakit
    """
    model = DiseaseCategory
    template_name = 'appsRLD/disease_list.html'
    context_object_name = 'diseases'
    login_url = '/login/'

    def get_queryset(self):
        return DiseaseCategory.objects.annotate(
            diagnosis_count=Count('diagnoses')
        ).order_by('name')


class DiseaseDetailView(DetailView):
    """
    Tampilkan detail informasi penyakit
    """
    model = DiseaseCategory
    template_name = 'appsRLD/disease_detail.html'
    context_object_name = 'disease'
    pk_url_kwarg = 'disease_id'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        disease = self.object

        context['diagnosis_count'] = disease.diagnoses.count()
        context['avg_confidence'] = disease.diagnoses.aggregate(
            avg_conf=Avg('max_confidence')
        )['avg_conf'] or 0
        context['recent_cases'] = disease.diagnoses.select_related(
            'image'
        ).order_by('-diagnosed_at')[:5]

        return context


# ========== STATISTICS ==========
class StatisticsDashboardView(TemplateView):
    """
    Dashboard statistik lengkap
    """
    template_name = 'appsRLD/statistics.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        stats = SystemStatistics.get_stats()
        stats.update_statistics()

        disease_distribution = DiagnosisResult.objects.values(
            'predicted_disease__display_name'
        ).annotate(count=Count('id')).order_by('-count')

        confidence_ranges = {
            'Sangat Tinggi (90-100%)': DiagnosisResult.objects.filter(max_confidence__gte=90).count(),
            'Tinggi (75-89%)': DiagnosisResult.objects.filter(max_confidence__gte=75, max_confidence__lt=90).count(),
            'Sedang (60-74%)': DiagnosisResult.objects.filter(max_confidence__gte=60, max_confidence__lt=75).count(),
            'Rendah (<60%)': DiagnosisResult.objects.filter(max_confidence__lt=60).count(),
        }

        six_months_ago = timezone.now() - timedelta(days=180)
        monthly_data = DiagnosisResult.objects.filter(
            diagnosed_at__gte=six_months_ago
        ).annotate(
            month=TruncMonth('diagnosed_at')
        ).values('month').annotate(count=Count('id')).order_by('month')

        monthly_diagnoses = [
            {
                'month': item['month'].strftime('%Y-%m') if item['month'] else 'Unknown',
                'count': item['count']
            }
            for item in monthly_data
        ]

        this_month = timezone.now().replace(day=1)
        top_diseases_month = DiagnosisResult.objects.filter(
            diagnosed_at__gte=this_month
        ).values('predicted_disease__display_name').annotate(
            count=Count('id')
        ).order_by('-count')[:5]

        total_feedback = DiagnosisResult.objects.exclude(is_correct__isnull=True).count()
        correct_predictions = DiagnosisResult.objects.filter(is_correct=True).count()
        accuracy_rate = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0

        context.update({
            'stats': stats,
            'disease_distribution': json.dumps(list(disease_distribution)),
            'confidence_ranges': confidence_ranges,
            'monthly_diagnoses': json.dumps(monthly_diagnoses),
            'top_diseases_month': top_diseases_month,
            'total_feedback': total_feedback,
            'accuracy_rate': accuracy_rate
        })
        return context


# ========== ABOUT ==========
# class AboutView(TemplateView):
#     """
#     Halaman tentang aplikasi dan metodologi
#     """
#     template_name = 'appsRLD/about.html'
#     login_url = '/login/'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['methodology'] = {
#             'preprocessing': [
#                 'Resizing (256x256)',
#                 'Grayscale Conversion',
#                 'Gaussian Filter (Noise Reduction)',
#                 'Normalisasi'
#             ],
#             'feature_extraction': [
#                 'GLCM (Gray Level Co-occurrence Matrix)',
#                 '6 Fitur: Contrast, Correlation, Energy, Homogeneity, Dissimilarity, ASM',
#                 '4 Sudut: 0°, 45°, 90°, 135°',
#                 'Total: 24 Fitur'
#             ],
#             'classification': [
#                 'Random Forest Classifier',
#                 'SMOTE untuk handling imbalanced data',
#                 '5 Kelas: Bacterial Blight, Rice Blast, Tungro, Healthy, Rice'
#             ]
#         }
#         context['dataset_info'] = {
#             'name': 'Rice Leaf and Crop Disease Detection Dataset',
#             'source': 'Mendeley Data',
#             'total_samples': '10,766 images',
#             'classes': 4
#         }
#         return context

class AboutView(LoginRequiredMixin, TemplateView):
    template_name = 'appsRLD/about.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['methodology'] = {
            'preprocessing': [
                {'icon': 'bi-arrows-angle-contract', 'title': 'Resizing', 'desc': 'Mengubah ukuran gambar menjadi 256×256 piksel untuk konsistensi input model.'},
                {'icon': 'bi-circle-half', 'title': 'Grayscale Conversion', 'desc': 'Mengkonversi gambar RGB ke grayscale untuk menyederhanakan analisis tekstur.'},
                {'icon': 'bi-wind', 'title': 'Gaussian Filter', 'desc': 'Mengurangi noise pada gambar menggunakan kernel 5×5 untuk hasil ekstraksi fitur yang lebih bersih.'},
                {'icon': 'bi-sliders', 'title': 'Normalisasi', 'desc': 'Menormalkan nilai piksel ke rentang 0–255 agar fitur GLCM lebih stabil.'},
            ],
            'feature_extraction': [
                {'icon': 'bi-grid-3x3', 'title': 'GLCM (24 Fitur)', 'desc': 'Gray Level Co-occurrence Matrix pada 4 sudut (0°, 45°, 90°, 135°): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.'},
                {'icon': 'bi-palette-fill', 'title': 'Color Features (39 Fitur)', 'desc': 'Fitur warna dari ruang warna HSV dan LAB: mean, std, skewness per channel + histogram BGR 8 bins.'},
                {'icon': 'bi-layout-wtf',  'title': 'LBP (29 Fitur)', 'desc': 'Local Binary Pattern dengan radius=3, n_points=24: histogram uniform LBP + statistik mean, std, entropy.'},
            ],
            'handling_imbalanced': [
                {'icon': 'bi-bezier2', 'title': 'BorderlineSMOTE', 'desc': 'Synthetic Minority Over-sampling Technique versi Borderline untuk menghasilkan sampel sintetis yang lebih representatif di batas keputusan.'},
                {'icon': 'bi-arrow-left-right', 'title': 'Data Augmentasi', 'desc': 'Flip horizontal/vertikal, rotasi 90°/180°, dan variasi brightness untuk kelas minoritas (Bacterial Blight & Tungro).'},
            ],
            'classification': [
                {'icon': 'bi-tree-fill', 'title': 'Random Forest', 'desc': 'Ensemble 500 decision trees dengan max_features=sqrt dan class_weight=balanced.'},
                {'icon': 'bi-diagram-3-fill', 'title': 'Extra Trees', 'desc': 'Extremely Randomized Trees dengan 800 estimators, lebih acak dari RF untuk mengurangi variance.'},
                {'icon': 'bi-graph-up-arrow', 'title': 'Gradient Boosting', 'desc': 'Sequential boosting dengan 400 estimators, learning_rate=0.05, dan max_depth=6.'},
                {'icon': 'bi-collection-fill', 'title': 'Voting Ensemble (Soft)', 'desc': 'Menggabungkan prediksi probabilitas dari RF + ET + GB menggunakan soft voting untuk akurasi optimal.'},
            ],
        }

        context['dataset_info'] = {
            'name': 'Rice Leaf and Crop Disease Detection Dataset',
            'source': 'Mendeley Data',
            'total_samples': '2,804',
            'augmented_samples': '~5,000+',
            'classes': 4,
            'class_list': [
                {'name': 'Bacterial Leaf Blight', 'count': '442', 'color': 'danger', 'icon': 'bi-bug-fill'},
                {'name': 'Rice Blast', 'count': '897', 'color': 'warning', 'icon': 'bi-virus'},
                {'name': 'Tungro', 'count': '537', 'color': 'info', 'icon': 'bi-virus2'},
                {'name': 'Healthy', 'count': '928', 'color': 'success', 'icon': 'bi-heart-fill'},
            ]
        }

        context['model_performance'] = {
            'accuracy': 93,
            'precision': 93,
            'recall': 93,
            'f1_score': 93,
            'per_class': [
                {'name': 'Bacterial Blight', 'accuracy': 89, 'color': 'danger'},
                {'name': 'Rice Blast', 'accuracy': 94, 'color': 'warning'},
                {'name': 'Tungro', 'accuracy': 91, 'color': 'info'},
                {'name': 'Healthy', 'accuracy': 98, 'color': 'success'},
            ]
        }

        context['tech_stack'] = [
            {'name': 'Django 5.2', 'icon': 'bi-server', 'color': 'success', 'desc': 'Web Framework'},
            {'name': 'Python 3.13', 'icon': 'bi-code-slash', 'color': 'primary', 'desc': 'Programming Language'},
            {'name': 'Scikit-learn', 'icon': 'bi-robot', 'color': 'warning', 'desc': 'Machine Learning'},
            {'name': 'OpenCV', 'icon': 'bi-camera-fill', 'color': 'info', 'desc': 'Image Processing'},
            {'name': 'PostgreSQL', 'icon': 'bi-database-fill', 'color': 'primary', 'desc': 'Database'},
            {'name': 'Bootstrap 5', 'icon': 'bi-bootstrap-fill', 'color': 'purple', 'desc': 'UI Framework'},
        ]

        context['training_scores'] = {
            'train_val': [
                {'label': 'Validation Accuracy', 'value': 84.76, 'color': 'primary'},
                {'label': 'Testing Accuracy',    'value': 93.00, 'color': 'success'},
                {'label': 'Testing Precision',   'value': 93.00, 'color': 'info'},
                {'label': 'Testing Recall',      'value': 93.00, 'color': 'warning'},
                {'label': 'Testing F1-Score',    'value': 93.00, 'color': 'danger'},
            ],
            'per_class_detail': [
                {'name': 'Bacterial Blight', 'precision': 89, 'recall': 87, 'f1': 88, 'support': 66,  'color': 'danger'},
                {'name': 'Rice Blast',       'precision': 94, 'recall': 93, 'f1': 93, 'support': 135, 'color': 'warning'},
                {'name': 'Tungro',           'precision': 91, 'recall': 90, 'f1': 91, 'support': 81,  'color': 'info'},
                {'name': 'Healthy',          'precision': 98, 'recall': 99, 'f1': 98, 'support': 139, 'color': 'success'},
            ],
            'split_info': {
                'total': 2804,
                'train': 1963,
                'val': 420,
                'test': 421,
                'smote_after': 2600,
            }
        }

        context['flow_steps'] = [
            {'number': 1, 'label': 'Input',            'desc': 'Gambar daun padi (upload / kamera)'},
            {'number': 2, 'label': 'Preprocessing',    'desc': 'Resize → Grayscale → Gaussian → Normalize'},
            {'number': 3, 'label': 'Ekstraksi Fitur',  'desc': 'GLCM + Color (HSV/LAB) + LBP = 92 fitur'},
            {'number': 4, 'label': 'Klasifikasi',      'desc': 'Voting Ensemble (RF + ET + GB)'},
            {'number': 5, 'label': 'Output',           'desc': 'Jenis penyakit + confidence score'},
        ]

        return context


# ========== REGISTER ==========
class RegisterView(View):
    """
    Halaman registrasi user baru
    """
    template_name = 'appsRLD/register.html'

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('appsRLD:home')
        return render(request, self.template_name, {'form': RegisterForm()})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('appsRLD:home')

        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(
                request,
                f"Selamat datang, {user.first_name or user.username}! Akun Anda berhasil dibuat."
            )
            return redirect('appsRLD:login')

        messages.error(request, "Registrasi gagal. Periksa kembali form Anda.")
        return render(request, self.template_name, {'form': form})


# ========== LOGIN ==========
class LoginView(View):
    """
    Halaman login user
    """
    template_name = 'appsRLD/login.html'

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('appsRLD:home')
        return render(request, self.template_name, {'form': LoginForm(request)})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('appsRLD:home')

        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(
                request,
                f"Selamat datang kembali, {user.first_name or user.username}!"
            )
            next_url = request.GET.get('next', 'appsRLD:home')
            return redirect(next_url)

        messages.error(request, "Username atau password salah.")
        return render(request, self.template_name, {'form': form})


# ========== LOGOUT ==========
class LogoutView(LoginRequiredMixin, View):
    """
    Logout user
    """
    login_url = '/login/'

    def get(self, request):
        username = request.user.first_name or request.user.username
        logout(request)
        messages.success(request, f"Sampai jumpa, {username}! Anda telah berhasil logout.")
        return redirect('appsRLD:login')


# ========== PROFILE ==========
class ProfileView(LoginRequiredMixin, View):
    """
    Halaman profil user dengan riwayat diagnosis miliknya
    """
    template_name = 'appsRLD/profile.html'
    login_url = '/login/'

    def get(self, request):
        user_diagnoses = DiagnosisResult.objects.select_related(
            'image', 'predicted_disease'
        ).filter(image__user=request.user).order_by('-diagnosed_at')

        total_diagnoses = user_diagnoses.count()
        avg_confidence = user_diagnoses.aggregate(
            avg=Avg('max_confidence')
        )['avg'] or 0

        user_disease_dist = user_diagnoses.values(
            'predicted_disease__display_name'
        ).annotate(count=Count('id')).order_by('-count')

        paginator = Paginator(user_diagnoses, 10)
        page_obj = paginator.get_page(request.GET.get('page'))

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'total_diagnoses': total_diagnoses,
            'avg_confidence': avg_confidence,
            'user_disease_dist': user_disease_dist,
        })


# ========== DELETE DIAGNOSIS ==========
class DeleteDiagnosisView(LoginRequiredMixin, View):
    """
    Hapus diagnosis (untuk admin/user pemilik)
    """
    login_url = '/login/'

    def get(self, request, diagnosis_id):
        diagnosis = get_object_or_404(DiagnosisResult, id=diagnosis_id)

        if request.user.is_staff or (
            diagnosis.image.user and diagnosis.image.user == request.user
        ):
            image = diagnosis.image
            diagnosis.delete()
            if image.diagnoses.count() == 0:
                image.delete()
            messages.success(request, "Diagnosis berhasil dihapus.")
        else:
            messages.error(request, "Anda tidak memiliki izin untuk menghapus diagnosis ini.")

        return redirect('appsRLD:history')


# ========== API QUICK PREDICT ==========
class ApiQuickPredictView(View):
    """
    API endpoint untuk quick prediction (AJAX)
    """
    @method_decorator(require_http_methods(["POST"]))
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def post(self, request):
        if pipeline.model is None:
            return JsonResponse({'success': False, 'error': 'Model not loaded'}, status=500)

        if 'image' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No image provided'}, status=400)

        try:
            img = Image.open(request.FILES['image'])
            result = pipeline.predict_single_image(img)

            return JsonResponse({
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities'],
                'processing_time': result['total_time']
            })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)