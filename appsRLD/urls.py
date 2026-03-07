from django.urls import path
from . import views

app_name = 'appsRLD'

urlpatterns = [
    # Login sebagai halaman utama
    path('', views.LoginView.as_view(), name='home_redirect'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('profile/', views.ProfileView.as_view(), name='profile'),

    # Home (setelah login)
    path('home/', views.HomeView.as_view(), name='home'),

    # Upload & Diagnosis
    path('upload/', views.UploadAndDiagnoseView.as_view(), name='upload'),
    path('result/<int:diagnosis_id>/', views.DiagnosisResultView.as_view(), name='result'),

    # Camera Capture & Diagnosis
    # path('camera/', views.CameraCaptureDiagnoseView.as_view(), name='camera_capture'),
    path('camera-capture/', views.CameraCaptureDiagnoseView.as_view(), name='camera_capture'),

    # History
    path('history/', views.DiagnosisHistoryView.as_view(), name='history'),
    path('diagnosis/delete/<int:diagnosis_id>/', views.DeleteDiagnosisView.as_view(), name='delete_diagnosis'),
    path('history/clear-all/', views.ClearAllHistoryView.as_view(), name='clear_all_history'),

    # Disease Information
    path('diseases/', views.DiseaseListView.as_view(), name='disease_list'),
    path('diseases/<int:disease_id>/', views.DiseaseDetailView.as_view(), name='disease_detail'),

    # Statistics
    path('statistics/', views.StatisticsDashboardView.as_view(), name='statistics'),

    # About
    path('about/', views.AboutView.as_view(), name='about'),

    # API
    path('api/predict/', views.ApiQuickPredictView.as_view(), name='api_predict'),

    # ===== ADMIN PANEL =====
    path('admin-panel/', views.AdminDashboardView.as_view(), name='admin_dashboard'),
    path('admin-panel/users/', views.AdminUserListView.as_view(), name='admin_user_list'),
    path('admin-panel/users/<int:user_id>/', views.AdminUserDetailView.as_view(), name='admin_user_detail'),
    path('admin-panel/users/<int:user_id>/toggle/', views.AdminToggleUserView.as_view(), name='admin_toggle_user'),
    path('admin-panel/diagnoses/', views.AdminDiagnosisListView.as_view(), name='admin_diagnosis_list'),
    path('admin-panel/diagnoses/delete/<int:diagnosis_id>/', views.AdminDeleteDiagnosisView.as_view(), name='admin_delete_diagnosis'),
]