# Sapi Weight Predictor - Kalkulator Jagal

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi estimasi hasil potong dan kalkulator jagal.

## Fokus Tampilan

Sidebar tetap sederhana dan fokus pada input utama:

- Jenis ternak
- Bangsa ternak
- Jenis kelamin
- Lingkar dada
- Panjang badan
- Tombol hitung

Fitur lanjutan ditempatkan di tab hasil agar tampilan tidak terlalu penuh.

## Tab Hasil

Setelah tombol **Hitung Berat Badan** ditekan, hasil tampil dalam tab:

```text
⚖️ Hitung Berat Badan
🎯 Simulasi Target Berat
💰 Estimasi Ekonomi
📊 Biaya & Keuntungan
🔪 Kalkulator Jagal
```

## Fitur Utama

- Prediksi berat badan sapi, kambing, dan domba.
- Rumus menyesuaikan jenis dan bangsa ternak.
- Status ukuran ternak.
- Margin error prediksi berat.
- BCS / kondisi tubuh ternak.
- Skor akurasi input pengukuran.
- Simulasi target berat.
- Estimasi karkas, non-karkas, dan daging.
- Estimasi ekonomi berdasarkan jenis, bangsa, dan kelas pasar ternak.
- Estimasi biaya pemeliharaan, total modal, keuntungan, dan ROI.
- Kalkulator jagal:
  - Harga beli ternak
  - Biaya pemotongan
  - Biaya transportasi
  - Biaya tenaga kerja
  - Biaya es/penyimpanan
  - Biaya sewa/retribusi
  - Estimasi omzet daging
  - Estimasi omzet tulang & lemak
  - Estimasi omzet non-karkas
  - Total modal jagal
  - Profit jagal
  - ROI jagal
  - Harga beli impas
  - Harga beli maksimal sesuai target margin
  - Rekomendasi keputusan: Layak Dibeli, Perlu Negosiasi, atau Berisiko Rugi
- Rekomendasi otomatis.
- Download laporan PDF.
- Riwayat perhitungan.
- Upload data banyak ternak melalui CSV/Excel.
- Download template CSV.

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

- Nilai non-karkas pada kalkulator jagal memakai harga rata-rata gabungan.
- Untuk transaksi aktual, harga kulit, kepala, kaki, jeroan, dan komponen lain sebaiknya disesuaikan dengan pasar setempat.
- BCS adalah penilaian sederhana kondisi tubuh, bukan diagnosis kesehatan.
- Simulasi target berat adalah pendekatan matematis dari rumus aplikasi, bukan prediksi pertumbuhan biologis.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
