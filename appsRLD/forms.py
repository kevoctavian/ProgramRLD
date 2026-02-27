"""
Forms untuk Upload Gambar dan Diagnosis Penyakit Daun Padi
"""

from django import forms
from .models import RiceLeafImage, DiagnosisResult, DiseaseCategory
from PIL import Image
import os
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class ImageUploadForm(forms.ModelForm):
    """
    Form untuk upload gambar daun padi
    """
    
    class Meta:
        model = RiceLeafImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'imageInput'
            })
        }
        labels = {
            'image': 'Pilih Gambar Daun Padi'
        }
        help_texts = {
            'image': 'Format: JPG, JPEG, PNG. Maksimal 5MB.'
        }
    
    def clean_image(self):
        """
        Validasi gambar yang diupload
        """
        image = self.cleaned_data.get('image')
        
        if image:
            # Cek ukuran file (max 5MB)
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError("Ukuran file terlalu besar. Maksimal 5MB.")
            
            # Cek format file
            ext = os.path.splitext(image.name)[1].lower()
            valid_extensions = ['.jpg', '.jpeg', '.png']
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f"Format file tidak valid. Gunakan: {', '.join(valid_extensions)}"
                )
            
            # Cek apakah file benar-benar gambar
            try:
                img = Image.open(image)
                img.verify()
                
                # Reset file pointer setelah verify
                image.seek(0)
                
                # Cek dimensi minimal
                width, height = img.size
                if width < 50 or height < 50:
                    raise forms.ValidationError("Gambar terlalu kecil. Minimal 50x50 piksel.")
                
            except Exception as e:
                raise forms.ValidationError(f"File bukan gambar yang valid: {str(e)}")
        
        return image


class FeedbackForm(forms.ModelForm):
    """
    Form untuk user memberikan feedback pada hasil diagnosis
    """
    
    class Meta:
        model = DiagnosisResult
        fields = ['is_correct', 'actual_disease', 'notes']
        widgets = {
            'is_correct': forms.Select(
                choices=[
                    (None, '-- Pilih --'),
                    (True, 'Prediksi Benar'),
                    (False, 'Prediksi Salah')
                ],
                attrs={'class': 'form-select'}
            ),
            'actual_disease': forms.Select(attrs={'class': 'form-select'}),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Tambahkan catatan (opsional)...'
            })
        }
        labels = {
            'is_correct': 'Apakah Prediksi Benar?',
            'actual_disease': 'Penyakit Sebenarnya (jika salah)',
            'notes': 'Catatan Tambahan'
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make actual_disease required only if is_correct is False
        self.fields['actual_disease'].required = False
        self.fields['notes'].required = False


# class BatchUploadForm(forms.Form):
#     """
#     Form untuk upload multiple images sekaligus
#     """
#     images = forms.FileField(
#         widget=forms.FileInput(attrs={
#             'multiple': True,
#             'class': 'form-control',
#             'accept': 'image/*'
#         }),
#         label='Pilih Gambar (Multiple)',
#         help_text='Bisa pilih beberapa gambar sekaligus. Max 5MB per file.'
#     )
    
#     def clean_images(self):
#         """
#         Validasi multiple images
#         """
#         files = self.files.getlist('images')
        
#         if not files:
#             raise forms.ValidationError("Pilih minimal 1 gambar.")
        
#         if len(files) > 20:
#             raise forms.ValidationError("Maksimal 20 gambar dalam 1 kali upload.")
        
#         valid_extensions = ['.jpg', '.jpeg', '.png']
        
#         for file in files:
#             # Cek ukuran
#             if file.size > 5 * 1024 * 1024:
#                 raise forms.ValidationError(
#                     f"File {file.name} terlalu besar. Maksimal 5MB per file."
#                 )
            
#             # Cek format
#             ext = os.path.splitext(file.name)[1].lower()
#             if ext not in valid_extensions:
#                 raise forms.ValidationError(
#                     f"File {file.name} format tidak valid. Gunakan: {', '.join(valid_extensions)}"
#                 )
        
#         return files


class SearchForm(forms.Form):
    """
    Form untuk search/filter riwayat diagnosis
    """
    disease = forms.ModelChoiceField(
        queryset=DiseaseCategory.objects.all(),
        required=False,
        empty_label="-- Semua Penyakit --",
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Filter Penyakit'
    )
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        }),
        label='Dari Tanggal'
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        }),
        label='Sampai Tanggal'
    )
    
    min_confidence = forms.FloatField(
        required=False,
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Min confidence (%)',
            'step': '0.1'
        }),
        label='Confidence Minimal (%)'
    )
    
    search_query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Cari berdasarkan nama file...'
        }),
        label='Cari'
    )

class RegisterForm(UserCreationForm):
    """
    Form untuk registrasi user baru
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Masukkan email Anda'
        }),
        label='Email'
    )
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nama depan'
        }),
        label='Nama Depan'
    )
    last_name = forms.CharField(
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nama belakang (opsional)'
        }),
        label='Nama Belakang'
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Buat username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Buat password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Konfirmasi password'
        })
        # Terjemahkan label
        self.fields['username'].label = 'Username'
        self.fields['password1'].label = 'Password'
        self.fields['password2'].label = 'Konfirmasi Password'

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email ini sudah digunakan.")
        return email


class LoginForm(AuthenticationForm):
    """
    Form untuk login user
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Masukkan username'
        })
        self.fields['password'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Masukkan password'
        })
        self.fields['username'].label = 'Username'
        self.fields['password'].label = 'Password'