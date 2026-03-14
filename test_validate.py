"""
Script untuk mengecek apakah validate_rice_leaf di ml_pipeline.py
sudah menggunakan versi terbaru (v4 dengan unnatural color detection).

Jalankan dari root project Django:
    python test_validate.py

Atau dari folder appsRLD:
    python ../test_validate.py
"""

import os
import sys
import cv2
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProgramRLD.settings')
import django
django.setup()

from appsRLD.ml_pipeline import RiceDiseasePipeline

pipeline = RiceDiseasePipeline()

# ============================================================
# BUAT GAMBAR TEST SINTETIS
# ============================================================

def make_solid_green():
    """Poster/solid hijau — harus INVALID"""
    img = np.zeros((256, 256, 3), np.uint8)
    img[:] = [30, 180, 80]
    cv2.putText(img, "PANITIA", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 255), 4)
    return img

def make_table():
    """Tabel/spreadsheet — harus INVALID"""
    img = np.ones((256, 256, 3), np.uint8) * 230
    for i in range(0, 256, 18):
        cv2.line(img, (0, i), (256, i), (40, 40, 40), 1)
        cv2.line(img, (i, 0), (i, 256), (40, 40, 40), 1)
    return img

def make_frog():
    """Katak hijau + orange + biru — harus INVALID"""
    img = np.zeros((256, 256, 3), np.uint8)
    img[:] = [40, 130, 50]
    for _ in range(20):
        x, y = np.random.randint(30, 220, 2)
        cv2.circle(img, (x, y), np.random.randint(5, 15), [200, 50, 20], -1)  # biru
    cv2.ellipse(img, (60, 200), (35, 20), 0, 0, 360, [20, 100, 240], -1)   # kaki orange
    cv2.ellipse(img, (196, 200), (35, 20), 0, 0, 360, [20, 100, 240], -1)
    return img

def make_rice_leaf_sim():
    """Simulasi daun padi hijau dengan venasi — harus VALID"""
    img = np.zeros((256, 256, 3), np.uint8)
    img[:] = [30, 140, 60]
    # Tambah noise tekstur (simulasi venasi)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Venasi tengah
    cv2.line(img, (128, 0), (128, 256), [20, 100, 40], 2)
    return img

# ============================================================
# JALANKAN TEST
# ============================================================

test_cases = [
    # (gambar, nama, expected_valid)
    (make_solid_green(), "Poster hijau + teks",     False),
    (make_table(),       "Tabel spreadsheet",        False),
    (make_frog(),        "Katak (hijau+biru+orange)",False),
    (make_rice_leaf_sim(),"Simulasi daun padi",      True),
]

# Cek apakah ml_pipeline punya validate_rice_leaf
has_validate = hasattr(pipeline, 'validate_rice_leaf')
print("=" * 60)
print("CEK VERSI validate_rice_leaf")
print("=" * 60)

if not has_validate:
    print("❌ GAGAL: pipeline.validate_rice_leaf tidak ditemukan!")
    print("   Pastikan fungsi validate_rice_leaf sudah ditambahkan ke")
    print("   class RiceDiseasePipeline di ml_pipeline.py")
    sys.exit(1)

# Cek fitur baru ada di return value
img_test = make_solid_green()
result = pipeline.validate_rice_leaf(img_test)

print("\n📋 Field yang dikembalikan validate_rice_leaf:")
for k, v in result.items():
    print(f"   {k}: {v}")

# Cek field kunci versi terbaru
required_fields = ['unnatural_ratio', 'non_bg_ratio', 'sat_mean', 'local_std']
missing = [f for f in required_fields if f not in result]

print()
if missing:
    print(f"❌ VERSI LAMA — field berikut TIDAK ADA: {missing}")
    print("   Silakan update validate_rice_leaf dengan versi FINAL terbaru.")
else:
    print("✅ VERSI TERBARU — semua field kunci ditemukan!")

# ============================================================
# TEST AKURASI VALIDASI
# ============================================================
print("\n" + "=" * 60)
print("TEST AKURASI VALIDASI")
print("=" * 60)

passed = 0
failed = 0

for img, name, expected in test_cases:
    result = pipeline.validate_rice_leaf(img)
    is_valid = result['is_valid']
    score    = result['score']
    ok       = is_valid == expected
    status   = "✅ PASS" if ok else "❌ FAIL"
    expected_str = "VALID" if expected else "INVALID"
    actual_str   = "VALID" if is_valid else "INVALID"

    print(f"{status}  [{name}]")
    print(f"       expected={expected_str}  got={actual_str}  score={score}")
    if result['reasons']:
        print(f"       reasons={result['reasons']}")

    if ok:
        passed += 1
    else:
        failed += 1

print()
print(f"Hasil: {passed}/{passed+failed} test passed")
if failed == 0:
    print("🎉 Semua test PASS — validasi sudah menggunakan versi terbaru!")
else:
    print("⚠️  Ada test yang FAIL — validasi perlu dicek ulang.")