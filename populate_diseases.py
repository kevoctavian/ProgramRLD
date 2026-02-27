from appsRLD.models import DiseaseCategory

# Data penyakit padi
diseases_data = [
    {
        'name': 'bacterial_blight',
        'display_name': 'Bacterial Leaf Blight',
        'description': '''Bacterial Leaf Blight adalah penyakit bakteri yang disebabkan oleh Xanthomonas oryzae pv. oryzae. 
        Penyakit ini merupakan salah satu penyakit paling destruktif pada tanaman padi di seluruh dunia.''',
        'symptoms': '''- Lesi memanjang berwarna hijau kelabu hingga putih pada daun
        - Garis-garis berair (water-soaked) di sepanjang tepi daun
        - Daun menguning dan mengering dari ujung
        - Pada infeksi parah, seluruh daun bisa layu dan mati
        - Sering muncul eksudat bakteri berwarna kuning pada pagi hari
        ''',
        'causes': '''Penyebab: Bakteri Xanthomonas oryzae pv. oryzae
        Penyebaran:
        - Air hujan dan irigasi
        - Angin yang membawa droplet air
        - Alat pertanian yang terkontaminasi
        - Sisa tanaman yang terinfeksi
        Kondisi yang memicu:
        - Kelembaban tinggi (>70%)
        - Suhu 25-30°C
        - Luka pada tanaman (dari serangga, angin, dll)
        ''',
        'treatment': '''Pengendalian & Pencegahan:
        1. Gunakan varietas tahan (varietas lokal atau hibrida tahan BB)
        2. Gunakan benih bersertifikat dan sehat
        3. Perendaman benih dengan bakterisida sebelum tanam
        4. Pengaturan jarak tanam yang baik untuk sirkulasi udara
        5. Pemupukan berimbang (hindari nitrogen berlebih)
        6. Drainase yang baik untuk mengurangi kelembaban
        7. Rotasi tanaman
        8. Musnahkan sisa tanaman yang terinfeksi
        9. Aplikasi bakterisida berbasis tembaga saat gejala awal
        10. Sanitasi alat pertanian
        ''',
        'severity_level': 3
    },
    {
        'name': 'rice_blast',
        'display_name': 'Rice Blast',
        'description': '''Rice Blast atau Blas adalah penyakit jamur yang disebabkan oleh Pyricularia oryzae (Magnaporthe oryzae). 
        Ini adalah penyakit paling merusak pada tanaman padi di seluruh dunia.''',
        'symptoms': '''- Bercak-bercak kecil berbentuk belah ketupat (diamond-shaped) pada daun
        - Bercak berwarna coklat dengan bagian tengah abu-abu atau putih
        - Tepi bercak berwarna coklat tua hingga hitam
        - Pada serangan berat, beberapa bercak bergabung
        - Daun menguning dan mengering
        - Dapat menyerang leher malai (neck blast) menyebabkan gabah hampa
        - Bercak juga bisa muncul di batang dan bulir padi
        ''',
        'causes': '''Penyebab: Jamur Pyricularia oryzae (Magnaporthe oryzae)
        Penyebaran:
        - Spora jamur terbawa angin dan percikan air hujan
        - Sisa-sisa tanaman yang terinfeksi
        - Benih yang terkontaminasi
        Kondisi yang memicu:
        - Kelembaban tinggi (>90%) dengan embun pada malam hari
        - Suhu 20-30°C
        - Pemupukan nitrogen berlebihan
        - Tanaman terlalu rapat
        - Kekeringan diikuti kelembaban tinggi
        ''',
        'treatment': '''Pengendalian & Pencegahan:
        1. Tanam varietas tahan blast
        2. Gunakan benih sehat dan bersertifikat
        3. Perlakuan benih dengan fungisida sistemik
        4. Pengaturan jarak tanam yang tepat (25x25 cm)
        5. Pemupukan berimbang (hindari N berlebih, tambah K dan Si)
        6. Pengairan berselang (intermittent irrigation)
        7. Sanitasi lahan - musnahkan jerami dan gulma
        8. Monitoring rutin sejak fase vegetatif
        9. Aplikasi fungisida (Trikasiklazol, Isoprotiolan) saat gejala awal
        10. Rotasi fungisida untuk mencegah resistensi
        11. Aplikasi silika untuk memperkuat dinding sel tanaman
        ''',
        'severity_level': 3
    },
    {
        'name': 'tungro',
        'display_name': 'Tungro',
        'description': '''Tungro adalah penyakit virus pada padi yang disebabkan oleh kombinasi dua virus: 
        Rice Tungro Spherical Virus (RTSV) dan Rice Tungro Bacilliform Virus (RTBV). 
        Penyakit ini ditularkan oleh wereng hijau (Nephotettix virescens).''',
        'symptoms': '''- Daun muda berwarna kuning atau orange kekuningan
        - Pertumbuhan tanaman terhambat (stunting)
        - Jumlah anakan berkurang drastis
        - Daun lebih pendek dan sempit dari normal
        - Perubahan warna dimulai dari ujung daun
        - Tanaman tampak kerdil dibanding tanaman sehat
        - Pembentukan malai terlambat atau tidak terbentuk sama sekali
        - Gabah hampa atau tidak terisi penuh
        - Pada serangan dini, tanaman bisa mati
        ''',
        'causes': '''Penyebab: Kombinasi 2 virus
        - Rice Tungro Spherical Virus (RTSV)
        - Rice Tungro Bacilliform Virus (RTBV)
        
        Vektor penyakit:
        - Wereng hijau (Nephotettix virescens, N. nigropictus)
        - Wereng mengisap cairan tanaman yang terinfeksi
        - Virus berkembang di tubuh wereng
        - Ditularkan saat wereng mengisap tanaman sehat
        
        Faktor pemicu:
        - Populasi wereng hijau tinggi
        - Penanaman tidak serempak
        - Tersedia tanaman inang alternatif (rumput-rumputan)
        - Musim hujan dengan kelembaban tinggi
        ''',
        'treatment': '''Pengendalian & Pencegahan:
        1. Gunakan varietas tahan tungro (Inpari, Ciherang, dll)
        2. PENTING: Cabut dan musnahkan tanaman terinfeksi segera
        3. Tanam serempak dalam radius 500m untuk memutus siklus virus
        4. Pergiliran varietas setiap musim tanam
        5. Pengendalian wereng hijau (vektor):
           - Insektisida sistemik pada fase vegetatif awal
           - Monitoring populasi wereng dengan perangkap cahaya
           - Aplikasi insektisida saat populasi >2 ekor/rumpun
        6. Hindari penanaman padi terus-menerus (ratoon)
        7. Bersihkan gulma dan rumput liar di sekitar sawah
        8. Gunakan benih dari tanaman sehat
        9. Periode bera minimal 2 minggu antar musim tanam
        10. Hindari pemupukan nitrogen berlebihan
        
        CATATAN: Tidak ada pestisida yang dapat membunuh virus!
        Pengendalian fokus pada vektor (wereng) dan tanaman terinfeksi.
        ''',
        'severity_level': 3
    },
    {
        'name': 'healthy',
        'display_name': 'Healthy (Sehat)',
        'description': '''Daun padi yang sehat, tidak menunjukkan gejala penyakit. 
        Daun berwarna hijau segar, tegak, dan tidak ada bercak atau lesi.''',
        'symptoms': '''Ciri-ciri daun sehat:
        - Warna hijau segar merata
        - Permukaan daun mulus tanpa bercak
        - Tekstur daun tegak dan kuat
        - Tidak ada perubahan warna (menguning, coklat, putih)
        - Tidak ada lesi atau nekrosis
        - Pertumbuhan normal sesuai fase
        - Tidak ada tanda-tanda serangan hama atau penyakit
        ''',
        'causes': '''Kondisi optimal untuk pertumbuhan sehat:
        - Varietas unggul dan benih berkualitas
        - Pemupukan berimbang (N, P, K, Mikronutrien)
        - Pengairan yang cukup dan teratur
        - Pengendalian hama dan penyakit preventif
        - Sanitasi lahan yang baik
        - pH tanah optimal (5.5-7.0)
        - Cahaya matahari cukup
        - Drainase baik
        ''',
        'treatment': '''Pemeliharaan untuk menjaga kesehatan tanaman:
        1. Monitoring rutin kondisi tanaman
        2. Pemupukan sesuai rekomendasi dan fase pertumbuhan
        3. Pengairan teratur (genangan 5-10 cm fase vegetatif)
        4. Pengendalian gulma secara berkala
        5. Aplikasi pestisida preventif bila diperlukan
        6. Sanitasi lahan dan alat pertanian
        7. Rotasi tanaman untuk menjaga kesuburan tanah
        8. Penggunaan pupuk organik untuk kesehatan tanah
        9. Sistem drainase yang baik
        10. Jarak tanam optimal untuk sirkulasi udara
        ''',
        'severity_level': 0
    },
]

print("=" * 70)
print("MENGISI DATA KATEGORI PENYAKIT PADI")
print("=" * 70)

created_count = 0
updated_count = 0

for data in diseases_data:
    disease, created = DiseaseCategory.objects.update_or_create(
        name=data['name'],
        defaults={
            'display_name': data['display_name'],
            'description': data['description'],
            'symptoms': data['symptoms'],
            'causes': data['causes'],
            'treatment': data['treatment'],
            'severity_level': data['severity_level']
        }
    )
    
    if created:
        created_count += 1
        print(f"✓ Created: {disease.display_name}")
    else:
        updated_count += 1
        print(f"↻ Updated: {disease.display_name}")

print("=" * 70)
print(f"SELESAI! Created: {created_count}, Updated: {updated_count}")
print("=" * 70)
print("\nData kategori penyakit berhasil diisi!")
print("Anda sekarang bisa login ke admin panel untuk melihat data.")