# Sapi Weight Predictor - Weight Focus Tabs

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan.

## Fokus Tampilan Baru

Versi ini dibuat lebih fokus pada **hitung berat badan**:

- Sidebar hanya berisi input utama:
  - Jenis ternak
  - Bangsa ternak
  - Jenis kelamin
  - Lingkar dada
  - Panjang badan
  - Tombol hitung
- Hasil utama berada di tab **Hitung Berat Badan**.
- Estimasi harga dipindahkan ke tab **Estimasi Ekonomi**.
- Biaya, profit, dan ROI dipindahkan ke tab **Biaya & Keuntungan**.

## Fitur Utama

- Prediksi berat badan sapi, kambing, dan domba.
- Rumus menyesuaikan jenis dan bangsa ternak.
- Status ukuran ternak.
- Margin error prediksi berat.
- Estimasi karkas, non-karkas, dan daging.
- Estimasi ekonomi berdasarkan jenis, bangsa, dan kelas pasar ternak.
- Estimasi biaya pemeliharaan, total modal, keuntungan, dan ROI.
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

## Cara Menjalankan Lokal

Mac/Linux:

```bash
cd sapi_weight_predictor_streamlit_weight_focus_tabs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Windows:

```bash
cd sapi_weight_predictor_streamlit_weight_focus_tabs
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

## Catatan

- Harga default adalah acuan awal dan tetap bisa diedit manual sesuai daerah.
- Kelas pasar bersifat estimasi cepat, bukan penilaian resmi.
- Margin error dipakai agar hasil tidak dianggap sebagai angka pasti.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
