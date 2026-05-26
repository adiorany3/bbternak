# Sapi Weight Predictor

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi estimasi karkas, non-karkas, daging, serta visualisasi data.

## File Utama

Gunakan file berikut sebagai **main file** saat deploy di Streamlit Online:

```text
sapi_weight_predictor.py
```

## Isi Paket

```text
sapi_weight_predictor_streamlit/
├── sapi_weight_predictor.py      # file utama aplikasi Streamlit
├── requirements.txt              # dependency Python untuk Streamlit Online
├── README.md                     # panduan penggunaan dan deploy
├── .gitignore
├── .streamlit/
│   └── config.toml               # konfigurasi tema Streamlit
├── assets/
│   ├── lingkar_dada.png
│   └── panjang_badan.png
├── run_app.sh                    # menjalankan lokal di Mac/Linux
├── run_app.bat                   # menjalankan lokal di Windows
└── original_source.txt           # cadangan kode sumber awal
```

## Cara Menjalankan di Laptop/PC

```bash
cd sapi_weight_predictor_streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Untuk Windows:

```bash
cd sapi_weight_predictor_streamlit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run sapi_weight_predictor.py
```

Atau jalankan cepat:

```bash
./run_app.sh
```

Windows:

```bat
run_app.bat
```

## Cara Upload ke GitHub

1. Extract file ZIP ini.
2. Buka folder `sapi_weight_predictor_streamlit`.
3. Upload semua isi folder ke repository GitHub.
4. Pastikan file `sapi_weight_predictor.py` dan `requirements.txt` berada di root repository, bukan tersimpan di dalam folder tambahan yang bertingkat.

Struktur yang benar di GitHub:

```text
repository-anda/
├── sapi_weight_predictor.py
├── requirements.txt
├── README.md
├── .streamlit/
└── assets/
```

## Cara Deploy di Streamlit Online

1. Buka Streamlit Community Cloud.
2. Pilih **New app**.
3. Pilih repository GitHub Anda.
4. Pada bagian **Main file path**, isi:

```text
sapi_weight_predictor.py
```

5. Klik **Deploy**.

## Catatan Penting untuk Streamlit Online

- Jangan ubah nama `requirements.txt`.
- Jangan hapus folder `assets`, karena gambar panduan pengukuran diambil dari folder tersebut.
- Jika aplikasi gagal membaca gambar, aplikasi tetap berjalan karena sudah disiapkan fallback pesan teks.
- File utama sudah disesuaikan menjadi `sapi_weight_predictor.py`.

## Catatan Perbaikan

Versi ini sudah ditambahkan beberapa perbaikan aman:

1. Hasil perhitungan tetap tersimpan memakai `st.session_state`.
2. Path gambar dibuat lebih aman dengan folder `assets/`.
3. Ada validasi sederhana jika hasil berat badan terlihat ekstrem.
4. Perhitungan persentase non-karkas dibuat aman dari pembagian dengan nol.
5. Nama file utama disesuaikan untuk deploy Streamlit Online.

## Catatan Akurasi

Rumus, konstanta, faktor koreksi bangsa, dan persentase karkas tetap perlu divalidasi ulang dengan rujukan akademik/lapangan sebelum aplikasi digunakan untuk keputusan jual beli, penelitian, atau produksi.
