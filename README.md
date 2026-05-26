# Sapi Weight Predictor - Systematic UI

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi analisis berdasarkan **jenis ternak + bangsa ternak**.

## Fokus Versi Ini

Versi ini dirapikan agar lebih sistematis dan mudah dipahami. Urutan penggunaan dibuat jelas:

```text
1. Input data utama di sidebar
2. Hitung berat badan
3. Baca hasil utama
4. Lanjutkan ke analisis target/ekonomi/jagal/blantik
5. Cek insight
6. Simpan PDF/CSV atau gunakan mode banyak ternak
```


## Penyempurnaan Target Berat

Tab **2️⃣ Target Berat** sudah disesuaikan berdasarkan **jenis ternak + bangsa ternak**.

Perubahan:
- Contoh target tidak lagi hanya memakai persentase umum dari berat saat ini.
- Target dihitung dari rentang normal lingkar dada dan panjang badan setiap bangsa ternak.
- Aplikasi menampilkan contoh:
  - Target Ringan
  - Target Standar
  - Target Optimal
  - Target Maksimal Normal
- Setiap target menampilkan:
  - Target berat
  - Estimasi lingkar dada
  - Estimasi panjang badan
  - Segmen pasar yang cocok
  - Catatan penggunaan
- Pengguna tetap bisa mengubah angka target secara manual.

## Penyempurnaan Struktur

- Panduan pengukuran dipindahkan ke expander agar halaman utama tidak terlalu panjang.
- Tab hasil diberi nomor urut agar alur baca lebih jelas.
- Detail teknis, hasil potong, visualisasi, dan laporan PDF masuk ke expander khusus.
- Riwayat dan upload banyak ternak ditempatkan sebagai bagian arsip.
- Instruksi mode banyak ternak dibuat lebih ringkas dan bertahap.
- Header menggunakan workflow ringkas: Input → Hitung → Baca Hasil → Analisis → Simpan.
- Desain tetap adaptif untuk light mode dan dark mode.


## Fitur Baru: Generator Prompt AI

Aplikasi sekarang memiliki tab **8️⃣ Prompt AI**.

Fitur ini menyusun prompt otomatis dari hasil perhitungan, sehingga peternak dapat menyalinnya ke AI lain untuk mendapatkan analisis lanjutan.

Jenis prompt:
- **Peternak**: fokus pada bobot, BCS, pakan, pemeliharaan, dan target berat.
- **Jagal**: fokus pada karkas, daging, susut, omzet, biaya, profit, dan harga beli maksimal.
- **Blantik**: fokus pada harga beli, harga jual ulang, margin, daya jual, segmentasi pembeli, dan strategi negosiasi.
- **Analisis Lengkap**: menggabungkan sudut pandang peternak, jagal, dan blantik.

Output:
- Prompt siap salin.
- Download prompt dalam format `.txt`.
- Panduan cara memakai prompt di AI lain.

## Tab Hasil

Setelah tombol **Hitung Berat Badan** ditekan, hasil tampil dalam tab:

```text
1️⃣ Berat & Akurasi
2️⃣ Target Berat
3️⃣ Ekonomi Ternak
4️⃣ Biaya & Profit
5️⃣ Jagal
6️⃣ Blantik
7️⃣ Insight
8️⃣ Prompt AI
```

## Bagian Arsip

Di bagian akhir aplikasi terdapat:

```text
📋 Riwayat & Unduhan
📤 Mode Banyak Ternak
```

## Penyesuaian Berdasarkan Jenis dan Bangsa

Aplikasi memakai profil bisnis tiap bangsa ternak untuk:
- Posisi pasar
- Pembeli potensial
- Kacamata jagal
- Kacamata blantik
- Strategi umum
- Risiko khusus
- Likuiditas pasar
- Kesesuaian untuk jagal
- Kesesuaian untuk penggemukan
- Faktor premium harga

## File Utama untuk Streamlit Online

Gunakan file berikut sebagai **Main file path**:

```text
sapi_weight_predictor.py
```

## Struktur Folder

```text
repository-anda/
├── sapi_weight_predictor.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── assets/
    ├── karkas.jpeg
    ├── lingkar_dada.png
    └── panjang_badan.png
```

## Cara Deploy ke Streamlit Community Cloud

1. Upload semua isi folder ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih **New app**.
4. Pilih repository GitHub.
5. Isi **Main file path**:

```text
sapi_weight_predictor.py
```

6. Klik **Deploy**.

## Catatan

- Hasil adalah estimasi, bukan pengganti timbangan ternak.
- Profil bangsa adalah alat bantu analisis pasar, bukan keputusan mutlak.
- Harga lokal, musim, lokasi, umur, kesehatan, dan permintaan pasar tetap perlu dicek langsung.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
