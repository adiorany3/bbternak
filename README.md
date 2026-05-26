# Sapi Weight Predictor - BCS, Target Berat, dan Skor Akurasi

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan.

## Fokus Tampilan

Sidebar tetap dibuat sederhana dan fokus pada input utama:

- Jenis ternak
- Bangsa ternak
- Jenis kelamin
- Lingkar dada
- Panjang badan
- Tombol hitung

Fitur lanjutan ditempatkan di tab hasil agar tampilan tidak terlalu penuh.

## Fitur Utama

- Prediksi berat badan sapi, kambing, dan domba.
- Rumus menyesuaikan jenis dan bangsa ternak.
- Status ukuran ternak.
- Margin error prediksi berat.
- BCS / kondisi tubuh ternak.
- Skor akurasi input pengukuran.
- Simulasi target berat:
  - Target berat badan
  - Estimasi lingkar dada target
  - Estimasi panjang badan target
  - Status realistis/tidaknya target
- Estimasi karkas, non-karkas, dan daging.
- Estimasi ekonomi berdasarkan jenis, bangsa, dan kelas pasar ternak.
- Estimasi biaya pemeliharaan, total modal, keuntungan, dan ROI.
- Rekomendasi otomatis.
- Download laporan PDF.
- Riwayat perhitungan.
- Upload data banyak ternak melalui CSV/Excel.
- Download template CSV.

## Tab Hasil

Setelah tombol **Hitung Berat Badan** ditekan, hasil tampil dalam tab:

```text
⚖️ Hitung Berat Badan
🎯 Simulasi Target Berat
💰 Estimasi Ekonomi
📊 Biaya & Keuntungan
```

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

## Cara Menjalankan Lokal

Mac/Linux:

```bash
cd sapi_weight_predictor_streamlit_bcs_target_accuracy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Windows:

```bash
cd sapi_weight_predictor_streamlit_bcs_target_accuracy
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

## Catatan

- BCS adalah penilaian sederhana kondisi tubuh, bukan diagnosis kesehatan.
- Skor akurasi input membantu membaca kualitas data pengukuran.
- Simulasi target berat adalah pendekatan matematis dari rumus aplikasi, bukan prediksi pertumbuhan biologis.
- Harga default adalah acuan awal dan tetap bisa diedit manual sesuai daerah.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
