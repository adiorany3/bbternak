# Sapi Weight Predictor - Insight Blantik

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi kalkulator jagal, insight analisis, dan insight blantik ternak.

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
📈 Insight Analisis
🤝 Insight Blantik
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
  - Omzet daging
  - Omzet tulang & lemak
  - Omzet non-karkas
  - Harga beli impas
  - Harga beli maksimal sesuai target margin
  - Rekomendasi keputusan jagal
- Insight Analisis:
  - Efisiensi karkas
  - Daging terhadap bobot hidup
  - Risiko utama
  - Sensitivitas harga jual
  - Sensitivitas susut daging
  - Struktur omzet
  - Checklist keputusan
- Insight Blantik:
  - Estimasi harga jual kembali
  - Margin bersih blantik
  - ROI blantik
  - Harga ideal beli
  - Harga maksimal beli
  - Harga impas
  - Skor daya jual
  - Segmentasi calon pembeli
  - Risiko transaksi
  - Strategi jual: jual cepat, tahan, penggemukan, atau jangan deal
  - Checklist tindakan transaksi
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

- Insight blantik adalah alat bantu tawar-menawar dan jual ulang, bukan keputusan mutlak.
- Tetap cek fisik ternak, umur, kesehatan, surat/kepemilikan, dan harga pasar setempat sebelum transaksi.
- Nilai non-karkas pada kalkulator jagal memakai harga rata-rata gabungan.
- BCS adalah penilaian sederhana kondisi tubuh, bukan diagnosis kesehatan.
- Simulasi target berat adalah pendekatan matematis dari rumus aplikasi, bukan prediksi pertumbuhan biologis.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
