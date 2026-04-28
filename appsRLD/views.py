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
from django.db.models import Case, Count, Avg, Q, IntegerField, Max, When
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
from django.db.models.functions import TruncDate, TruncMonth
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
class HomeView(LoginRequiredMixin, TemplateView):
    template_name = 'appsRLD/home.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        try:
            from .models import ModelTrainingHistory
            latest_training = ModelTrainingHistory.objects.filter(
                is_active=True
            ).order_by('-trained_at').first()
            model_accuracy = latest_training.accuracy if latest_training else 0
        except Exception:
            model_accuracy = 0

        # Base queryset per user
        if self.request.user.is_staff:
            base_qs = DiagnosisResult.objects.select_related('image', 'predicted_disease')
        else:
            base_qs = DiagnosisResult.objects.select_related(
                'image', 'predicted_disease'
            ).filter(image__user=self.request.user)

        recent_diagnoses = base_qs.order_by('-diagnosed_at')[:6]

        disease_distribution = base_qs.filter(
            predicted_disease__isnull=False
        ).values(
            'predicted_disease__display_name'
        ).annotate(count=Count('id')).order_by('-count')

        # Statistik manual per user
        from django.db.models import Avg
        total_images = base_qs.values('image').distinct().count()
        total_diagnoses = base_qs.count()
        avg_confidence = base_qs.aggregate(
            avg=Avg('max_confidence')
        )['avg'] or 0
        avg_processing_time = base_qs.aggregate(
            avg=Avg('total_time')
        )['avg'] or 0

        context.update({
            'model_accuracy': model_accuracy,
            'recent_diagnoses': recent_diagnoses,
            'disease_distribution': list(disease_distribution),
            'model_loaded': pipeline.model is not None,
            # Statistik per user
            'total_images': total_images,
            'total_diagnoses': total_diagnoses,
            'avg_confidence': round(avg_confidence, 2),
            'avg_processing_time': round(avg_processing_time, 4),
            # Per penyakit
            'total_bacterial_blight': base_qs.filter(
                predicted_disease__name='bacterial_blight'
            ).count(),
            'total_rice_blast': base_qs.filter(
                predicted_disease__name='rice_blast'
            ).count(),
            'total_tungro': base_qs.filter(
                predicted_disease__name='tungro'
            ).count(),
            'total_healthy': base_qs.filter(
                predicted_disease__name='healthy'
            ).count(),
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

                if not result.get('is_valid_leaf', True):
                    rice_image.image.delete(save=False)
                    rice_image.delete()
                    reasons = result.get('validation', {}).get('reasons', [])
                    reason_str = ' '.join(reasons).lower()
                    if any(k in reason_str for k in ['dokumen', 'kertas', 'teks', 'document', 'low_sat', 'garis lurus', 'hitam putih']):
                        rejection_type = 'document'
                    elif any(k in reason_str for k in ['poster', 'grafis', 'dominant_hue', 'mser', 'desain']):
                        rejection_type = 'poster'
                    elif any(k in reason_str for k in ['hewan', 'animal', 'katak', 'orange', 'biru jenuh', 'tidak wajar']):
                        rejection_type = 'animal'
                    elif any(k in reason_str for k in ['grayscale', 'abu', 'monokrom']):
                        rejection_type = 'grayscale'
                    else:
                        rejection_type = 'unknown'
                    from django.urls import reverse as _reverse
                    return redirect(f"{_reverse('appsRLD:upload')}?rejected=1&type={rejection_type}")

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
                from django.urls import reverse as _reverse
                return redirect(f"{_reverse('appsRLD:upload')}?rejected=1&type=unknown")

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

            # Validasi daun padi
            if not result.get('is_valid_leaf', True):
                rice_image.image.delete(save=False)
                rice_image.delete()
                reasons = result.get('validation', {}).get('reasons', [])
                reason_str = ' '.join(reasons).lower()
                if any(k in reason_str for k in ['dokumen', 'kertas', 'teks', 'low_sat', 'garis lurus', 'hitam putih']):
                    rejection_type = 'document'
                elif any(k in reason_str for k in ['poster', 'grafis', 'dominant_hue', 'mser']):
                    rejection_type = 'poster'
                elif any(k in reason_str for k in ['hewan', 'animal', 'katak', 'orange', 'tidak wajar']):
                    rejection_type = 'animal'
                elif any(k in reason_str for k in ['grayscale', 'abu', 'monokrom']):
                    rejection_type = 'grayscale'
                else:
                    rejection_type = 'unknown'
                from django.urls import reverse as _reverse
                return redirect(f"{_reverse('appsRLD:upload')}?rejected=1&type={rejection_type}")

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
            from django.urls import reverse as _reverse
            return redirect(f"{_reverse('appsRLD:upload')}?rejected=1&type=unknown")

# ========== DIAGNOSIS RESULT ==========
class DiagnosisResultView(LoginRequiredMixin, View):
    template_name = 'appsRLD/result.html'
    login_url = '/login/'

    def get(self, request, diagnosis_id):
        diagnosis = get_object_or_404(
            DiagnosisResult.objects.select_related(
                'image', 'predicted_disease', 'actual_disease'
            ),
            id=diagnosis_id
        )

        # Cek kepemilikan
        is_owner = (
            diagnosis.image.user and
            diagnosis.image.user == request.user
        )
        if not (request.user.is_staff or is_owner):
            messages.error(request, "Anda tidak memiliki akses ke diagnosis ini.")
            return redirect('appsRLD:history')

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

        # Cek kepemilikan
        is_owner = (
            diagnosis.image.user and
            diagnosis.image.user == request.user
        )
        if not (request.user.is_staff or is_owner):
            messages.error(request, "Anda tidak memiliki akses ke diagnosis ini.")
            return redirect('appsRLD:history')

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
        # Filter berdasarkan user
        if request.user.is_staff:
            diagnoses = DiagnosisResult.objects.select_related(
                'image', 'predicted_disease'
            ).order_by('-diagnosed_at')
        else:
            diagnoses = DiagnosisResult.objects.select_related(
                'image', 'predicted_disease'
            ).filter(image__user=request.user).order_by('-diagnosed_at')

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
class ClearAllHistoryView(LoginRequiredMixin, View):
    login_url = '/login/'

    def post(self, request):
        confirm = request.POST.get('confirm', '')

        if confirm != 'HAPUS':
            messages.error(request, "Konfirmasi gagal. Ketik 'HAPUS' untuk menghapus semua data.")
            return redirect('appsRLD:history')

        try:
            if request.user.is_staff:
                diagnosis_count = DiagnosisResult.objects.count()
                DiagnosisResult.objects.all().delete()
                RiceLeafImage.objects.all().delete()
            else:
                # Hapus hanya milik user yang sedang login
                diagnosis_count = DiagnosisResult.objects.filter(
                    image__user=request.user
                ).count()
                DiagnosisResult.objects.filter(
                    image__user=request.user
                ).delete()
                RiceLeafImage.objects.filter(
                    user=request.user
                ).delete()

            stats = SystemStatistics.get_stats()
            stats.update_statistics()

            messages.success(
                request,
                f"✓ Berhasil menghapus {diagnosis_count} diagnosis dari riwayat Anda."
            )
        except Exception as e:
            messages.error(request, f"✗ Error: {str(e)}")

        return redirect('appsRLD:history')

# ========== DISEASE INFO ==========
class DiseaseListView(LoginRequiredMixin, ListView):
    model = DiseaseCategory
    template_name = 'appsRLD/disease_list.html'
    context_object_name = 'diseases'
    login_url = '/login/'

    def get_queryset(self):
        user = self.request.user

        ordering = Case(
            When(name='healthy', then=1),
            default=0,
            output_field=IntegerField()
        )

        if user.is_staff:
            return DiseaseCategory.objects.annotate(
                diagnosis_count=Count('diagnoses'),
                order=ordering
            ).order_by('order', 'name')
        else:
            return DiseaseCategory.objects.annotate(
                diagnosis_count=Count(
                    'diagnoses',
                    filter=Q(diagnoses__image__user=user)
                ),
                order=ordering
            ).order_by('order', 'name')


class DiseaseDetailView(LoginRequiredMixin, DetailView):
    model = DiseaseCategory
    template_name = 'appsRLD/disease_detail.html'
    context_object_name = 'disease'
    pk_url_kwarg = 'disease_id'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        disease = self.object
        user = self.request.user

        if user.is_staff:
            # Admin lihat semua kasus
            diagnoses = disease.diagnoses
        else:
            # User hanya lihat kasus miliknya
            diagnoses = disease.diagnoses.filter(image__user=user)

        context['diagnosis_count'] = diagnoses.count()
        context['avg_confidence'] = diagnoses.aggregate(
            avg_conf=Avg('max_confidence')
        )['avg_conf'] or 0
        context['recent_cases'] = diagnoses.select_related(
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

        # Base queryset per user
        if self.request.user.is_staff:
            base_qs = DiagnosisResult.objects
        else:
            base_qs = DiagnosisResult.objects.filter(
                image__user=self.request.user
            )

        from django.db.models import Avg
        # Hitung statistik manual per user
        total_images = base_qs.values('image').distinct().count()
        total_diagnoses = base_qs.count()
        avg_confidence = base_qs.aggregate(avg=Avg('max_confidence'))['avg'] or 0
        avg_processing_time = base_qs.aggregate(
                avg=Avg('total_time')
            )['avg'] or 0

        disease_distribution = base_qs.filter(
            predicted_disease__isnull=False
        ).values(
            'predicted_disease__display_name'
        ).annotate(count=Count('id')).order_by('-count')

        confidence_ranges = {
            'Sangat Tinggi (90-100%)': base_qs.filter(max_confidence__gte=90).count(),
            'Tinggi (75-89%)': base_qs.filter(
                max_confidence__gte=75, max_confidence__lt=90
            ).count(),
            'Sedang (60-74%)': base_qs.filter(
                max_confidence__gte=60, max_confidence__lt=75
            ).count(),
            'Rendah (<60%)': base_qs.filter(max_confidence__lt=60).count(),
        }

        six_months_ago = timezone.now() - timedelta(days=180)
        monthly_data = base_qs.filter(
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
        top_diseases_month = base_qs.filter(
            diagnosed_at__gte=this_month
        ).values('predicted_disease__display_name').annotate(
            count=Count('id')
        ).order_by('-count')[:5]

        total_feedback = base_qs.exclude(is_correct__isnull=True).count()
        correct_predictions = base_qs.filter(is_correct=True).count()
        accuracy_rate = (
            correct_predictions / total_feedback * 100
        ) if total_feedback > 0 else 0

        context.update({
            'disease_distribution': json.dumps(list(disease_distribution)),
            'confidence_ranges': confidence_ranges,
            'monthly_diagnoses': json.dumps(monthly_diagnoses),
            'top_diseases_month': top_diseases_month,
            'total_feedback': total_feedback,
            'accuracy_rate': accuracy_rate,
            # Statistik per user (gantikan stats global)
            'total_images': total_images,
            'total_diagnoses': total_diagnoses,
            'avg_confidence': round(avg_confidence, 2),
            'avg_processing_time': round(avg_processing_time, 4),
            'total_bacterial_blight': base_qs.filter(
                predicted_disease__name='bacterial_blight'
            ).count(),
            'total_rice_blast': base_qs.filter(
                predicted_disease__name='rice_blast'
            ).count(),
            'total_tungro': base_qs.filter(
                predicted_disease__name='tungro'
            ).count(),
            'total_healthy': base_qs.filter(
                predicted_disease__name='healthy'
            ).count(),
        })
        return context


# ========== ABOUT ==========
# class AboutView(LoginRequiredMixin, TemplateView):
#     template_name = 'appsRLD/about.html'
#     login_url = '/login/'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)

#         context['methodology'] = {
#             'preprocessing': [
#                 {'icon': 'bi-arrows-angle-contract', 'title': 'Resizing', 'desc': 'Mengubah ukuran gambar menjadi 256×256 piksel untuk konsistensi input model.'},
#                 {'icon': 'bi-circle-half', 'title': 'Grayscale Conversion', 'desc': 'Mengkonversi gambar RGB ke grayscale untuk menyederhanakan analisis tekstur.'},
#                 {'icon': 'bi-wind', 'title': 'Gaussian Filter', 'desc': 'Mengurangi noise pada gambar menggunakan kernel 5×5 untuk hasil ekstraksi fitur yang lebih bersih.'},
#                 {'icon': 'bi-sliders', 'title': 'Normalisasi', 'desc': 'Menormalkan nilai piksel ke rentang 0–255 agar fitur GLCM lebih stabil.'},
#             ],
#             'feature_extraction': [
#                 {'icon': 'bi-grid-3x3', 'title': 'GLCM (24 Fitur)', 'desc': 'Gray Level Co-occurrence Matrix pada 4 sudut (0°, 45°, 90°, 135°): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.'},
#                 {'icon': 'bi-palette-fill', 'title': 'Color Features (39 Fitur)', 'desc': 'Fitur warna dari ruang warna HSV dan LAB: mean, std, skewness per channel + histogram BGR 8 bins.'},
#                 {'icon': 'bi-layout-wtf',  'title': 'LBP (29 Fitur)', 'desc': 'Local Binary Pattern dengan radius=3, n_points=24: histogram uniform LBP + statistik mean, std, entropy.'},
#             ],
#             'handling_imbalanced': [
#                 {'icon': 'bi-bezier2', 'title': 'BorderlineSMOTE', 'desc': 'Synthetic Minority Over-sampling Technique versi Borderline untuk menghasilkan sampel sintetis yang lebih representatif di batas keputusan.'},
#                 {'icon': 'bi-arrow-left-right', 'title': 'Data Augmentasi', 'desc': 'Flip horizontal/vertikal, rotasi 90°/180°, dan variasi brightness untuk kelas minoritas (Bacterial Blight & Tungro).'},
#             ],
#             'classification': [
#                 {'icon': 'bi-tree-fill', 'title': 'Random Forest', 'desc': 'Ensemble 500 decision trees dengan max_features=sqrt dan class_weight=balanced.'},
#                 {'icon': 'bi-diagram-3-fill', 'title': 'Extra Trees', 'desc': 'Extremely Randomized Trees dengan 800 estimators, lebih acak dari RF untuk mengurangi variance.'},
#                 {'icon': 'bi-graph-up-arrow', 'title': 'Gradient Boosting', 'desc': 'Sequential boosting dengan 400 estimators, learning_rate=0.05, dan max_depth=6.'},
#                 {'icon': 'bi-collection-fill', 'title': 'Voting Ensemble (Soft)', 'desc': 'Menggabungkan prediksi probabilitas dari RF + ET + GB menggunakan soft voting untuk akurasi optimal.'},
#             ],
#         }

#         context['dataset_info'] = {
#             'name': 'Rice Leaf and Crop Disease Detection Dataset',
#             'source': 'Mendeley Data',
#             'total_samples': '2,804',
#             'augmented_samples': '~5,000+',
#             'classes': 4,
#             'class_list': [
#                 {'name': 'Bacterial Leaf Blight', 'count': '442', 'color': 'danger', 'icon': 'bi-bug-fill'},
#                 {'name': 'Rice Blast', 'count': '897', 'color': 'warning', 'icon': 'bi-virus'},
#                 {'name': 'Tungro', 'count': '537', 'color': 'info', 'icon': 'bi-virus2'},
#                 {'name': 'Healthy', 'count': '928', 'color': 'success', 'icon': 'bi-heart-fill'},
#             ]
#         }

#         context['model_performance'] = {
#             'accuracy': 93,
#             'precision': 93,
#             'recall': 93,
#             'f1_score': 93,
#             'per_class': [
#                 {'name': 'Bacterial Blight', 'accuracy': 89, 'color': 'danger'},
#                 {'name': 'Rice Blast', 'accuracy': 94, 'color': 'warning'},
#                 {'name': 'Tungro', 'accuracy': 91, 'color': 'info'},
#                 {'name': 'Healthy', 'accuracy': 98, 'color': 'success'},
#             ]
#         }

#         context['tech_stack'] = [
#             {'name': 'Django 5.2', 'icon': 'bi-server', 'color': 'success', 'desc': 'Web Framework'},
#             {'name': 'Python 3.13', 'icon': 'bi-code-slash', 'color': 'primary', 'desc': 'Programming Language'},
#             {'name': 'Scikit-learn', 'icon': 'bi-robot', 'color': 'warning', 'desc': 'Machine Learning'},
#             {'name': 'OpenCV', 'icon': 'bi-camera-fill', 'color': 'info', 'desc': 'Image Processing'},
#             {'name': 'PostgreSQL', 'icon': 'bi-database-fill', 'color': 'primary', 'desc': 'Database'},
#             {'name': 'Bootstrap 5', 'icon': 'bi-bootstrap-fill', 'color': 'purple', 'desc': 'UI Framework'},
#         ]

#         context['training_scores'] = {
#             'train_val': [
#                 {'label': 'Validation Accuracy', 'value': 84.76, 'color': 'primary'},
#                 {'label': 'Testing Accuracy',    'value': 93.00, 'color': 'success'},
#                 {'label': 'Testing Precision',   'value': 93.00, 'color': 'info'},
#                 {'label': 'Testing Recall',      'value': 93.00, 'color': 'warning'},
#                 {'label': 'Testing F1-Score',    'value': 93.00, 'color': 'danger'},
#             ],
#             'per_class_detail': [
#                 {'name': 'Bacterial Blight', 'precision': 89, 'recall': 87, 'f1': 88, 'support': 66,  'color': 'danger'},
#                 {'name': 'Rice Blast',       'precision': 94, 'recall': 93, 'f1': 93, 'support': 135, 'color': 'warning'},
#                 {'name': 'Tungro',           'precision': 91, 'recall': 90, 'f1': 91, 'support': 81,  'color': 'info'},
#                 {'name': 'Healthy',          'precision': 98, 'recall': 99, 'f1': 98, 'support': 139, 'color': 'success'},
#             ],
#             'split_info': {
#                 'total': 2804,
#                 'train': 1963,
#                 'val': 420,
#                 'test': 421,
#                 'smote_after': 2600,
#             }
#         }

#         context['flow_steps'] = [
#             {'number': 1, 'label': 'Input',            'desc': 'Gambar daun padi (upload / kamera)'},
#             {'number': 2, 'label': 'Preprocessing',    'desc': 'Resize → Grayscale → Gaussian → Normalize'},
#             {'number': 3, 'label': 'Ekstraksi Fitur',  'desc': 'GLCM + Color (HSV/LAB) + LBP = 92 fitur'},
#             {'number': 4, 'label': 'Klasifikasi',      'desc': 'Voting Ensemble (RF + ET + GB)'},
#             {'number': 5, 'label': 'Output',           'desc': 'Jenis penyakit + confidence score'},
#         ]
#         return context


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
            if request.user.is_staff:
                return redirect('appsRLD:admin_dashboard')
            return redirect('appsRLD:home')

        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(
                request,
                f"Selamat datang kembali, {user.first_name or user.username}!"
            )
            if user.is_staff:
                return redirect('appsRLD:admin_dashboard')
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
    login_url = '/login/'

    def get(self, request, diagnosis_id):
        diagnosis = get_object_or_404(DiagnosisResult, id=diagnosis_id)

        # Cek kepemilikan
        is_owner = (
            diagnosis.image.user and
            diagnosis.image.user == request.user
        )

        if not (request.user.is_staff or is_owner):
            messages.error(request, "Anda tidak memiliki izin untuk menghapus diagnosis ini.")
            return redirect('appsRLD:history')

        image = diagnosis.image
        diagnosis.delete()
        if image.diagnoses.count() == 0:
            image.delete()

        messages.success(request, "✓ Diagnosis berhasil dihapus.")
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

# ========== ADMIN MIXIN ==========
class AdminRequiredMixin(LoginRequiredMixin):
    """Hanya staff/admin yang boleh akses"""
    login_url = '/login/'

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        if not request.user.is_staff:
            messages.error(request, "Anda tidak memiliki akses ke halaman admin.")
            return redirect('appsRLD:home')
        return super().dispatch(request, *args, **kwargs)
    
class AboutView(AdminRequiredMixin, TemplateView):
    template_name = 'appsRLD/about.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        try:
            from appsRLD.models import ModelTrainingHistory
            latest_training = ModelTrainingHistory.objects.filter(is_active=True).first()
        except Exception:
            latest_training = None

        context['latest_training'] = latest_training

        # =====================================================================
        # Visualisasi
        # =====================================================================
        import os
        VIZ_DIR = os.path.join(settings.BASE_DIR, 'ml_models', 'visualizations')
        context['confusion_matrix_exists'] = os.path.exists(os.path.join(VIZ_DIR, 'confusion_matrix.png'))
        context['roc_curve_exists']        = os.path.exists(os.path.join(VIZ_DIR, 'roc_curve.png'))

        # =====================================================================
        # Dataset
        # =====================================================================
        context['dataset_info'] = {
            'name': 'Rice Leaf and Crop Disease Detection Dataset',
            'source': 'Mendeley Data',
            'total_samples': latest_training.total_samples if latest_training else '2,804',
            'augmented_samples': '~5,000+',
            'classes': 4,
            'class_list': [
                {'name': 'Bacterial Leaf Blight', 'count': '442', 'color': 'danger',  'icon': 'bi-bug-fill'},
                {'name': 'Rice Blast',            'count': '897', 'color': 'warning', 'icon': 'bi-virus'},
                {'name': 'Tungro',                'count': '537', 'color': 'info',    'icon': 'bi-virus2'},
                {'name': 'Healthy',               'count': '928', 'color': 'success', 'icon': 'bi-heart-fill'},
            ]
        }

        # =====================================================================
        # Performa Model
        # =====================================================================
        acc  = round(latest_training.accuracy,  1) if latest_training else 93
        prec = round(latest_training.precision, 1) if latest_training else 93
        rec  = round(latest_training.recall,    1) if latest_training else 93
        f1   = round(latest_training.f1_score,  1) if latest_training else 93

        # Per-class dari DB jika tersedia, fallback hardcoded
        per_class_detail = latest_training.get_per_class_list() if latest_training else [
            {'name': 'Bacterial Blight', 'precision': 89, 'recall': 87, 'f1': 88, 'support': 66,  'color': 'danger'},
            {'name': 'Rice Blast',       'precision': 94, 'recall': 93, 'f1': 93, 'support': 135, 'color': 'warning'},
            {'name': 'Tungro',           'precision': 91, 'recall': 90, 'f1': 91, 'support': 81,  'color': 'info'},
            {'name': 'Healthy',          'precision': 98, 'recall': 99, 'f1': 98, 'support': 139, 'color': 'success'},
        ]

        context['model_performance'] = {
            'accuracy':  acc,
            'precision': prec,
            'recall':    rec,
            'f1_score':  f1,
            # Per-class untuk progress bar akurasi — ambil f1 dari per_class_detail
            'per_class': [
                {
                    'name':     cls['name'],
                    'accuracy': cls['f1'],
                    'color':    cls['color'],
                }
                for cls in per_class_detail
            ] if per_class_detail else [
                {'name': 'Bacterial Blight', 'accuracy': 89, 'color': 'danger'},
                {'name': 'Rice Blast',       'accuracy': 94, 'color': 'warning'},
                {'name': 'Tungro',           'accuracy': 91, 'color': 'info'},
                {'name': 'Healthy',          'accuracy': 98, 'color': 'success'},
            ]
        }

        # =====================================================================
        # Training & Testing Scores
        # =====================================================================
        total   = latest_training.total_samples       if latest_training else 2804
        train_n = latest_training.training_samples    if latest_training else 1963
        val_n   = latest_training.validation_samples  if latest_training else 420
        test_n  = latest_training.test_samples        if latest_training else 421
        smote_n = latest_training.samples_after_smote if latest_training else 2600

        val_acc  = round(latest_training.val_accuracy, 2) if (latest_training and latest_training.val_accuracy) else 84.76
        test_acc = round(latest_training.accuracy,     2) if latest_training else 93.00
        test_pre = round(latest_training.precision,    2) if latest_training else 93.00
        test_rec = round(latest_training.recall,       2) if latest_training else 93.00
        test_f1  = round(latest_training.f1_score,     2) if latest_training else 93.00

        context['training_scores'] = {
            'train_val': [
                {'label': 'Akurasi Validasi',   'value': val_acc,  'color': 'primary'},
                {'label': 'Akurasi Pengujian',  'value': test_acc, 'color': 'success'},
                {'label': 'Presisi Pengujian',  'value': test_pre, 'color': 'info'},
                {'label': 'Recall Pengujian',   'value': test_rec, 'color': 'warning'},
                {'label': 'F1-Score Pengujian', 'value': test_f1,  'color': 'danger'},
            ],
            'per_class_detail': per_class_detail,
            'split_info': {
                'total':       total,
                'train':       train_n,
                'val':         val_n,
                'test':        test_n,
                'smote_after': smote_n,
            }
        }

        # =====================================================================
        # Tech Stack
        # =====================================================================
        context['tech_stack'] = [
            {'name': 'Django 5.2',   'icon': 'bi-server',         'color': 'success', 'desc': 'Web Framework'},
            {'name': 'Python 3.13',  'icon': 'bi-code-slash',     'color': 'primary', 'desc': 'Programming Language'},
            {'name': 'Scikit-learn', 'icon': 'bi-robot',          'color': 'warning', 'desc': 'Machine Learning'},
            {'name': 'OpenCV',       'icon': 'bi-camera-fill',    'color': 'info',    'desc': 'Image Processing'},
            {'name': 'PostgreSQL',   'icon': 'bi-database-fill',  'color': 'primary', 'desc': 'Database'},
            {'name': 'Bootstrap 5',  'icon': 'bi-bootstrap-fill', 'color': 'purple',  'desc': 'UI Framework'},
        ]

        # =====================================================================
        # Methodology
        # =====================================================================
        context['methodology'] = {
            'preprocessing': [
                {'icon': 'bi-arrows-angle-contract', 'color': '#3498db', 'title': 'Resizing',
                 'desc': 'Mengubah ukuran gambar menjadi 256x256 piksel untuk konsistensi input model.'},
                {'icon': 'bi-circle-half',           'color': '#9b59b6', 'title': 'Grayscale Conversion',
                 'desc': 'Mengkonversi gambar RGB ke grayscale untuk menyederhanakan analisis tekstur.'},
                {'icon': 'bi-wind',                  'color': '#1abc9c', 'title': 'Gaussian Filter',
                 'desc': 'Mengurangi noise pada gambar menggunakan kernel 5x5 untuk hasil ekstraksi fitur yang lebih bersih.'},
                {'icon': 'bi-sliders',               'color': '#e67e22', 'title': 'Normalisasi',
                 'desc': 'Menormalkan nilai piksel ke rentang 0-255 agar fitur GLCM lebih stabil.'},
            ],
            'feature_extraction': [
                {'icon': 'bi-grid-3x3',    'color': '#e74c3c', 'title': 'GLCM (24 Fitur)',
                 'desc': 'Gray Level Co-occurrence Matrix pada 4 sudut (0, 45, 90, 135 derajat): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.'},
                {'icon': 'bi-palette-fill','color': '#f39c12', 'title': 'Color Features (39 Fitur)',
                 'desc': 'Fitur warna dari ruang warna HSV dan LAB: mean, std, skewness per channel + histogram BGR 8 bins.'},
                {'icon': 'bi-layout-wtf',  'color': '#2ecc71', 'title': 'LBP (29 Fitur)',
                 'desc': 'Local Binary Pattern dengan radius=3, n_points=24: histogram uniform LBP + statistik mean, std, entropy.'},
            ],
            'handling_imbalanced': [
                {'icon': 'bi-bezier2',         'color': '#8e44ad', 'title': 'BorderlineSMOTE',
                 'desc': 'Synthetic Minority Over-sampling Technique versi Borderline untuk menghasilkan sampel sintetis yang lebih representatif di batas keputusan.'},
                {'icon': 'bi-arrow-left-right', 'color': '#16a085', 'title': 'Data Augmentasi',
                 'desc': 'Flip horizontal/vertikal, rotasi 90/180 derajat, dan variasi brightness untuk kelas minoritas (Bacterial Blight dan Tungro).'},
            ],
            'classification': [
                {'icon': 'bi-tree-fill', 'color': '#27ae60', 'title': 'Random Forest',
                 'desc': 'Ensemble learning berbasis decision tree dengan GridSearchCV untuk optimasi hyperparameter (n_estimators, max_depth, max_features). Menggunakan class_weight=balanced untuk menangani ketidakseimbangan kelas.'},
            ],
        }

        return context
    
# ========== ADMIN DASHBOARD ==========
class AdminDashboardView(AdminRequiredMixin, TemplateView):
    template_name = 'appsRLD/admin/dashboard.html'
 
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
 
        from django.contrib.auth.models import User
 
        total_users    = User.objects.filter(is_staff=False).count()
        active_users   = User.objects.filter(is_staff=False, is_active=True).count()
        inactive_users = User.objects.filter(is_staff=False, is_active=False).count()
        total_diagnoses = DiagnosisResult.objects.count()
        diagnoses_today = DiagnosisResult.objects.filter(diagnosed_at__date=timezone.now().date()).count()
        avg_confidence  = DiagnosisResult.objects.aggregate(avg=Avg('max_confidence'))['avg'] or 0
 
        disease_dist = DiagnosisResult.objects.filter(
            predicted_disease__isnull=False
        ).values('predicted_disease__display_name').annotate(
            count=Count('id')
        ).order_by('-count')
 
        recent_users = User.objects.filter(is_staff=False).order_by('-date_joined')[:5]
 
        recent_diagnoses = DiagnosisResult.objects.select_related(
            'image', 'image__user', 'predicted_disease'
        ).order_by('-diagnosed_at')[:10]
 
        # === Penyakit Terbanyak Bulan Ini ===
        this_month = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        top_diseases_month = list(
            DiagnosisResult.objects.filter(
                diagnosed_at__gte=this_month,
                predicted_disease__isnull=False
            ).values('predicted_disease__display_name', 'predicted_disease__name').annotate(
                count=Count('id')
            ).order_by('-count')[:5]
        )

        # === Akurasi Model ===
        try:
            from appsRLD.models import ModelTrainingHistory
            latest_training = ModelTrainingHistory.objects.filter(is_active=True).first()
            model_accuracy = round(latest_training.accuracy, 1) if latest_training else 0
        except Exception:
            model_accuracy = 0

        context.update({
            'total_users':          total_users,
            'active_users':         active_users,
            'inactive_users':       inactive_users,
            'total_diagnoses':      total_diagnoses,
            'diagnoses_today':      diagnoses_today,
            'avg_confidence':       round(avg_confidence, 2),
            'disease_dist':         json.dumps(list(disease_dist)),
            'recent_users':         recent_users,
            'recent_diagnoses':     recent_diagnoses,
            'top_diseases_month':   top_diseases_month,
            'model_accuracy':       model_accuracy,
            'total_bacterial_blight': DiagnosisResult.objects.filter(predicted_disease__name='bacterial_blight').count(),
            'total_rice_blast':       DiagnosisResult.objects.filter(predicted_disease__name='rice_blast').count(),
            'total_tungro':           DiagnosisResult.objects.filter(predicted_disease__name='tungro').count(),
            'total_healthy':          DiagnosisResult.objects.filter(predicted_disease__name='healthy').count(),
        })
        return context


# ========== ADMIN USER LIST ==========
class AdminUserListView(AdminRequiredMixin, View):
    template_name = 'appsRLD/admin/user_list.html'

    def get(self, request):
        from django.contrib.auth.models import User
        from django.db.models import Avg

        search = request.GET.get('search', '')
        status = request.GET.get('status', '')

        users = User.objects.filter(is_staff=False).annotate(
            diagnosis_count=Count('uploaded_images__diagnoses'),
            last_diagnosis=Max('uploaded_images__diagnoses__diagnosed_at')
        ).order_by('-date_joined')

        if search:
            users = users.filter(
                Q(username__icontains=search) |
                Q(first_name__icontains=search) |
                Q(last_name__icontains=search) |
                Q(email__icontains=search)
            )
        if status == 'active':
            users = users.filter(is_active=True)
        elif status == 'inactive':
            users = users.filter(is_active=False)

        paginator = Paginator(users, 15)
        page_obj  = paginator.get_page(request.GET.get('page'))

        return render(request, self.template_name, {
            'page_obj': page_obj,
            'search':   search,
            'status':   status,
            'total_users': users.count(),
        })


# ========== ADMIN USER DETAIL ==========
class AdminUserDetailView(AdminRequiredMixin, View):
    template_name = 'appsRLD/admin/user_detail.html'

    def get(self, request, user_id):
        from django.contrib.auth.models import User
        from django.db.models import Avg

        target_user = get_object_or_404(User, id=user_id, is_staff=False)

        diagnoses = DiagnosisResult.objects.select_related(
            'image', 'predicted_disease'
        ).filter(image__user=target_user).order_by('-diagnosed_at')

        stats = diagnoses.aggregate(
            avg_conf=Avg('max_confidence'),
            avg_time=Avg('total_time'),
        )

        disease_dist = diagnoses.filter(
            predicted_disease__isnull=False
        ).values('predicted_disease__display_name').annotate(
            count=Count('id')
        ).order_by('-count')

        paginator = Paginator(diagnoses, 10)
        page_obj  = paginator.get_page(request.GET.get('page'))

        return render(request, self.template_name, {
            'target_user':    target_user,
            'page_obj':       page_obj,
            'total_diagnoses': diagnoses.count(),
            'avg_confidence': round(stats['avg_conf'] or 0, 2),
            'avg_time':       round(stats['avg_time'] or 0, 4),
            'disease_dist':   list(disease_dist),
        })


# ========== ADMIN TOGGLE USER ==========
class AdminToggleUserView(AdminRequiredMixin, View):

    def post(self, request, user_id):
        from django.contrib.auth.models import User
        target_user = get_object_or_404(User, id=user_id, is_staff=False)
        target_user.is_active = not target_user.is_active
        target_user.save()

        status = "diaktifkan" if target_user.is_active else "dinonaktifkan"
        messages.success(
            request,
            f"Akun {target_user.username} berhasil {status}."
        )
        return redirect('appsRLD:admin_user_detail', user_id=user_id)


# ========== ADMIN DIAGNOSIS LIST ==========
class AdminDiagnosisListView(AdminRequiredMixin, View):
    template_name = 'appsRLD/admin/diagnosis_list.html'

    def get(self, request):
        search  = request.GET.get('search', '')
        disease = request.GET.get('disease', '')

        diagnoses = DiagnosisResult.objects.select_related(
            'image', 'image__user', 'predicted_disease'
        ).order_by('-diagnosed_at')

        if search:
            diagnoses = diagnoses.filter(
                Q(image__user__username__icontains=search) |
                Q(image__original_filename__icontains=search)
            )
        if disease:
            diagnoses = diagnoses.filter(predicted_disease__name=disease)

        paginator = Paginator(diagnoses, 15)
        page_obj  = paginator.get_page(request.GET.get('page'))

        disease_choices = DiseaseCategory.objects.all()

        return render(request, self.template_name, {
            'page_obj':        page_obj,
            'search':          search,
            'disease_filter':  disease,
            'disease_choices': disease_choices,
            'total_count':     diagnoses.count(),
        })


# ========== ADMIN DELETE DIAGNOSIS ==========
class AdminDeleteDiagnosisView(AdminRequiredMixin, View):

    def post(self, request, diagnosis_id):
        diagnosis = get_object_or_404(DiagnosisResult, id=diagnosis_id)
        image     = diagnosis.image
        diagnosis.delete()
        if image.diagnoses.count() == 0:
            image.delete()
        messages.success(request, "✓ Diagnosis berhasil dihapus.")
        return redirect('appsRLD:admin_diagnosis_list')