# Sapi Weight Predictor - Profit Final

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan.

## Fitur Utama

- Prediksi berat badan sapi, kambing, dan domba.
- Rumus menyesuaikan jenis dan bangsa ternak.
- Estimasi karkas, non-karkas, dan daging.
- Harga default berdasarkan jenis dan bangsa ternak.
- Kelas/kondisi pasar: Otomatis, Kelas A / Super, Kelas B / Normal, Kelas C / Kurus.
- Margin error prediksi berat.
- Estimasi nilai bobot hidup, karkas, dan daging.
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

## Cara Menjalankan Lokal

Mac/Linux:

```bash
cd sapi_weight_predictor_streamlit_profit_final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Windows:

```bash
cd sapi_weight_predictor_streamlit_profit_final
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
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

- Harga default adalah acuan awal dan tetap bisa diedit manual sesuai daerah.
- Kelas pasar bersifat estimasi cepat, bukan penilaian resmi.
- Margin error dipakai agar hasil tidak dianggap sebagai angka pasti.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
