# Sapi Weight Predictor - Breed Wide Perspective

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan, dilengkapi analisis berdasarkan **jenis ternak + bangsa ternak**.


## Penyempurnaan Desain

Versi ini sudah dilengkapi desain adaptif untuk **light mode** dan **dark mode**:

- Warna teks otomatis menyesuaikan mode tampilan.
- Background menggunakan palet lembut yang tetap kontras.
- Sidebar dibuat lebih bersih.
- Metric card lebih jelas dan konsisten.
- Tab dibuat seperti pill agar mudah dibaca.
- Tabel, expander, tombol, alert, dan footer dibuat adaptif.
- Footer tidak lagi memakai warna statis yang berisiko kurang terbaca di dark mode.

## Fokus Versi Ini

Versi ini memperluas sudut pandang aplikasi. Analisis tidak lagi hanya berdasarkan jenis ternak umum, tetapi juga disesuaikan dengan karakter **bangsa ternak**.

Contoh:
- Sapi Bali dibaca sebagai sapi lokal adaptif dan likuid untuk pasar rakyat/kurban.
- Sapi Limousin dan Simental dibaca sebagai sapi besar/premium dengan kebutuhan modal dan pembeli berbeda.
- Kambing Boer dibaca sebagai kambing pedaging premium.
- Domba Garut dibaca sebagai domba lokal bernilai tinggi untuk segmen premium/pejantan/kurban.
- Domba Texel dan Suffolk dibaca sebagai domba pedaging premium.

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

## Penyesuaian Berdasarkan Jenis dan Bangsa

Aplikasi sekarang memakai profil bisnis tiap bangsa ternak untuk:
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

## Fitur Utama

- Prediksi berat badan sapi, kambing, dan domba.
- Rumus menyesuaikan jenis dan bangsa ternak.
- Profil pasar berdasarkan jenis dan bangsa.
- Status ukuran ternak.
- Margin error prediksi berat.
- BCS / kondisi tubuh ternak.
- Skor akurasi input pengukuran.
- Simulasi target berat dengan catatan khusus bangsa.
- Estimasi karkas, non-karkas, dan daging.
- Estimasi ekonomi berdasarkan jenis, bangsa, kelas pasar, dan faktor premium.
- Estimasi biaya pemeliharaan, total modal, keuntungan, dan ROI.
- Kalkulator jagal berbasis profil bangsa:
  - Omzet daging
  - Omzet tulang & lemak
  - Omzet non-karkas
  - Kesesuaian jagal
  - Harga beli impas
  - Harga beli maksimal sesuai target margin
  - Rekomendasi keputusan jagal
- Insight Analisis:
  - Efisiensi karkas
  - Daging terhadap bobot hidup
  - Risiko khusus bangsa
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
  - Skor daya jual berbasis bangsa
  - Segmentasi calon pembeli berdasarkan bangsa
  - Risiko transaksi
  - Strategi jual: jual cepat, tahan, penggemukan, atau jangan deal
  - Checklist tindakan transaksi
- Riwayat perhitungan.
- Upload data banyak ternak melalui CSV/Excel.
- Download template CSV.
- Download laporan PDF.

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

- Profil bangsa adalah alat bantu analisis pasar, bukan keputusan mutlak.
- Harga lokal, musim, lokasi, umur, kesehatan, dan permintaan pasar tetap perlu dicek langsung.
- Untuk transaksi besar, tetap gunakan timbangan ternak terkalibrasi.
