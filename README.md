# Sapi Weight Predictor - Versi Terbaru

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi estimasi karkas, non-karkas, daging, estimasi nilai ekonomi, laporan PDF, riwayat, dan upload data massal.

## File Utama untuk Streamlit Online

Gunakan file berikut sebagai **Main file path**:

```text
sapi_weight_predictor.py
```

## Struktur Paket

```text
repository-anda/
├── sapi_weight_predictor.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── assets/
    ├── lingkar_dada.png
    ├── panjang_badan.png
    └── karkas.jpeg
```

## Cara Menjalankan Lokal

Mac/Linux:

```bash
cd sapi_weight_predictor_streamlit_latest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Windows:

```bash
cd sapi_weight_predictor_streamlit_latest
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

## Cara Deploy di Streamlit Community Cloud

1. Upload semua isi folder ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih **New app**.
4. Pilih repository GitHub Anda.
5. Isi **Main file path** dengan:

```text
sapi_weight_predictor.py
```

6. Klik **Deploy**.

## Catatan Penting

- Jangan hapus `requirements.txt`.
- Jangan hapus folder `assets`.
- Jika ingin mengganti gambar, gunakan nama file yang sama atau sesuaikan pemanggilan gambar pada kode.
- Harga default bersifat acuan dan tetap dapat diubah manual oleh pengguna.
- Hasil prediksi tetap estimasi, bukan pengganti timbangan ternak.
