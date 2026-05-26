#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediksi Berat Badan Ternak (Sapi, Kambing, Domba) menggunakan Rumus Formula

Aplikasi Streamlit untuk menghitung prediksi berat badan ternak berdasarkan 
lingkar dada dan panjang badan menggunakan rumus-rumus yang spesifik untuk
jenis dan bangsa ternak yang berbeda.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
from pathlib import Path

# Get current year for the footer
current_year = datetime.now().year


# Harga default terbaru yang dipakai aplikasi.
# Catatan: Harga bersifat acuan nasional/pasar dan tetap dapat diubah manual oleh pengguna.
LATEST_PRICE_DEFAULTS = {
    "Sapi": {
        "harga_bobot_hidup": 55000,
        "harga_karkas": 107000,
        "harga_daging": 150750,
        "label": "Acuan terbaru sapi: bobot hidup Rp55.000/kg, karkas Rp107.000/kg, daging sapi kualitas I Rp150.750/kg.",
        "source": "PIHPS/BI 25 Mei 2026; Bapanas Maret 2026; HAP Bapanas 2026."
    },
    "Kambing": {
        "harga_bobot_hidup": 90000,
        "harga_karkas": 135000,
        "harga_daging": 155000,
        "label": "Acuan pasar kambing terbaru: karkas sekitar Rp135.000/kg dan daging sekitar Rp155.000/kg.",
        "source": "Acuan pasar online/ritel Mei 2026; sesuaikan dengan harga daerah."
    },
    "Domba": {
        "harga_bobot_hidup": 95000,
        "harga_karkas": 135000,
        "harga_daging": 150000,
        "label": "Acuan pasar domba terbaru: karkas sekitar Rp135.000/kg dan daging sekitar Rp150.000/kg.",
        "source": "Acuan pasar online/ritel Mei 2026; sesuaikan dengan harga daerah."
    }
}

# Faktor penyesuaian harga berdasarkan bangsa ternak.
# Nilai ini dipakai untuk membuat default harga lebih spesifik per bangsa ternak,
# tetapi pengguna tetap dapat mengubah harga secara manual sesuai pasar daerah.
BREED_PRICE_FACTORS = {
    "Sapi": {
        "Sapi Bali": 1.00,
        "Sapi Madura": 0.96,
        "Sapi Limousin": 1.18,
        "Sapi Simental": 1.16,
        "Sapi Brahman": 1.10,
        "Sapi Peranakan Ongole (PO)": 1.02,
        "Sapi Friesian Holstein (FH)": 0.98,
        "Sapi Aceh": 0.94,
    },
    "Kambing": {
        "Kambing Kacang": 0.95,
        "Kambing Ettawa": 1.08,
        "Kambing Peranakan Ettawa (PE)": 1.04,
        "Kambing Boer": 1.15,
        "Kambing Jawarandu": 1.00,
        "Kambing Bligon": 0.98,
    },
    "Domba": {
        "Domba Ekor Tipis": 0.96,
        "Domba Ekor Gemuk": 1.02,
        "Domba Merino": 1.10,
        "Domba Garut": 1.08,
        "Domba Suffolk": 1.13,
        "Domba Texel": 1.15,
    },
}

# Faktor penyesuaian harga berdasarkan kelas/kondisi pasar ternak.
# Kelas ini bersifat estimasi cepat dan tetap bisa disesuaikan manual di sidebar.

BREED_BUSINESS_PROFILES = {
    "Sapi": {
        "Sapi Bali": {
            "market_position": "Sapi lokal adaptif, likuid di pasar tradisional, kuat untuk kurban dan penggemukan skala rakyat.",
            "primary_buyers": ["Pembeli kurban", "Peternak penggemukan", "Pedagang pasar hewan", "Jagal lokal"],
            "butcher_view": "Cocok untuk jagal lokal dan pasar daging segar; nilai jual terbantu oleh permintaan lokal yang stabil.",
            "trader_view": "Mudah diputar karena dikenal luas; margin sering berasal dari ketepatan beli dan kondisi tubuh.",
            "strategy": "Jual cepat jika BCS ideal; penggemukan singkat jika BCS masih kurus tetapi rangka normal.",
            "risks": ["Harga bisa sangat dipengaruhi musim kurban", "Bobot ekstrem belum tentu mudah diserap semua pembeli"],
            "liquidity_bonus": 8,
            "butcher_fit": 7,
            "fattening_fit": 8,
            "premium_factor": 1.00,
        },
        "Sapi Madura": {
            "market_position": "Sapi lokal dengan pasar kuat di wilayah tertentu, cocok untuk transaksi tradisional dan kurban.",
            "primary_buyers": ["Pembeli kurban", "Pedagang pasar hewan", "Peternak lokal", "Jagal lokal"],
            "butcher_view": "Cocok untuk potong lokal; perlu cermat pada bobot dan BCS agar nilai karkas tetap menarik.",
            "trader_view": "Daya jual baik di pasar yang mengenal sapi Madura; di luar wilayah kuat perlu strategi harga.",
            "strategy": "Fokus pada pembeli lokal/kurban; hindari beli terlalu mahal bila ukuran kecil.",
            "risks": ["Pasar cenderung regional", "Harga jual ulang bisa lebih sensitif lokasi"],
            "liquidity_bonus": 6,
            "butcher_fit": 6,
            "fattening_fit": 7,
            "premium_factor": 0.96,
        },
        "Sapi Limousin": {
            "market_position": "Sapi besar/premium, menarik untuk jagal, kurban premium, dan transaksi bernilai tinggi.",
            "primary_buyers": ["Jagal besar", "Pembeli kurban premium", "Pedagang besar", "Peternak penggemukan"],
            "butcher_view": "Potensi karkas dan daging tinggi; sangat menarik jika harga beli masih masuk batas margin.",
            "trader_view": "Nilai jual tinggi, tetapi modal besar; perlu pembeli yang tepat agar perputaran tidak lambat.",
            "strategy": "Cocok untuk jual premium atau jagal; pastikan margin aman karena modal tinggi.",
            "risks": ["Modal besar", "Jika harga beli terlalu tinggi, ruang nego mengecil", "Perlu pembeli kelas premium"],
            "liquidity_bonus": 7,
            "butcher_fit": 9,
            "fattening_fit": 8,
            "premium_factor": 1.18,
        },
        "Sapi Simental": {
            "market_position": "Sapi besar/premium dengan daya tarik tinggi untuk daging, kurban premium, dan jagal.",
            "primary_buyers": ["Jagal besar", "Pembeli kurban premium", "Pedagang besar", "Peternak penggemukan"],
            "butcher_view": "Potensi daging baik; cocok untuk jagal jika efisiensi karkas dan harga beli seimbang.",
            "trader_view": "Daya jual tinggi, tetapi harus diarahkan ke pembeli bermodal besar.",
            "strategy": "Jual ke segmen premium; gunakan batas harga maksimal agar tidak terjebak modal besar.",
            "risks": ["Butuh pasar premium", "Biaya tahan dan transport bisa besar"],
            "liquidity_bonus": 7,
            "butcher_fit": 9,
            "fattening_fit": 8,
            "premium_factor": 1.16,
        },
        "Sapi Brahman": {
            "market_position": "Sapi tipe besar dan tahan lingkungan; cocok untuk penggemukan dan pasar potong.",
            "primary_buyers": ["Jagal", "Feedlot/penggemukan", "Pedagang besar", "Peternak"],
            "butcher_view": "Menarik untuk potong jika bobot dan BCS cukup; cocok untuk pasar daging volume.",
            "trader_view": "Cocok untuk pembeli yang mencari performa dan bobot; perlu cek kondisi tubuh.",
            "strategy": "Tahan/penggemukan jika BCS belum ideal; jual cepat bila bobot dan harga sudah masuk.",
            "risks": ["Variasi kondisi tubuh bisa besar", "Perlu pakan baik jika ditahan"],
            "liquidity_bonus": 7,
            "butcher_fit": 8,
            "fattening_fit": 9,
            "premium_factor": 1.10,
        },
        "Sapi Peranakan Ongole (PO)": {
            "market_position": "Sapi lokal-kerja/potong yang dikenal luas; cocok untuk penggemukan dan pasar tradisional.",
            "primary_buyers": ["Peternak penggemukan", "Jagal lokal", "Pedagang pasar hewan", "Pembeli kurban"],
            "butcher_view": "Masuk untuk potong lokal; margin bergantung pada harga beli dan BCS.",
            "trader_view": "Cukup likuid dan fleksibel, baik untuk jual ulang jika harga beli tidak tinggi.",
            "strategy": "Cari margin dari beli cermat dan penggemukan; cocok untuk pasar menengah.",
            "risks": ["Jika BCS rendah, hasil daging bisa kurang optimal", "Harga premium tidak setinggi sapi besar impor"],
            "liquidity_bonus": 7,
            "butcher_fit": 7,
            "fattening_fit": 8,
            "premium_factor": 1.02,
        },
        "Sapi Friesian Holstein (FH)": {
            "market_position": "Sapi perah/afkir atau persilangan yang dapat masuk pasar potong, tetapi perlu cermat kualitas daging.",
            "primary_buyers": ["Jagal tertentu", "Pedagang pasar hewan", "Peternak tertentu"],
            "butcher_view": "Perlu teliti karena karakter perah bisa berbeda dari sapi potong; harga beli harus konservatif.",
            "trader_view": "Cocok untuk transaksi jika ada pembeli jelas; jangan terlalu agresif pada harga beli.",
            "strategy": "Utamakan jual cepat ke pembeli yang memang mencari FH/potong; hindari tahan lama.",
            "risks": ["Persepsi pasar berbeda dari sapi potong", "Kondisi afkir bisa menurunkan nilai"],
            "liquidity_bonus": 4,
            "butcher_fit": 5,
            "fattening_fit": 5,
            "premium_factor": 0.98,
        },
        "Sapi Aceh": {
            "market_position": "Sapi lokal kecil-adaptif dengan pasar kuat di wilayah tertentu.",
            "primary_buyers": ["Pedagang lokal", "Pembeli kurban lokal", "Peternak lokal", "Jagal lokal"],
            "butcher_view": "Cocok untuk potong lokal; hasil harus dibaca sesuai ukuran tubuh yang relatif kecil.",
            "trader_view": "Baik di pasar yang mengenal sapi Aceh; margin lebih aman jika biaya angkut rendah.",
            "strategy": "Fokus pasar lokal/regional; jangan membandingkan langsung dengan sapi besar premium.",
            "risks": ["Pasar regional", "Ukuran kecil dapat membatasi segmen pembeli tertentu"],
            "liquidity_bonus": 5,
            "butcher_fit": 5,
            "fattening_fit": 6,
            "premium_factor": 0.94,
        },
    },
    "Kambing": {
        "Kambing Kacang": {
            "market_position": "Kambing lokal kecil yang likuid untuk pasar tradisional, aqiqah, dan konsumsi rumah tangga.",
            "primary_buyers": ["Pembeli aqiqah", "Pedagang pasar", "Jagal kecil", "Peternak lokal"],
            "butcher_view": "Cocok untuk potong kecil/eceran; harga beli harus sebanding dengan bobot.",
            "trader_view": "Perputaran cepat, tetapi margin per ekor sering tidak besar.",
            "strategy": "Jual cepat dan main volume; hindari biaya tahan terlalu lama.",
            "risks": ["Bobot kecil membatasi omzet per ekor", "Margin tipis jika biaya angkut tinggi"],
            "liquidity_bonus": 8,
            "butcher_fit": 6,
            "fattening_fit": 6,
            "premium_factor": 0.95,
        },
        "Kambing Ettawa": {
            "market_position": "Kambing besar/perah dengan nilai lebih pada ukuran dan penampilan.",
            "primary_buyers": ["Peternak bibit/perah", "Pembeli kurban", "Pedagang premium", "Jagal tertentu"],
            "butcher_view": "Nilai potong ada, tetapi jangan abaikan nilai non-potong seperti bibit/perah jika kondisinya bagus.",
            "trader_view": "Bisa masuk segmen premium; perlu pembeli yang tepat agar harga optimal.",
            "strategy": "Cari pembeli spesifik, bukan hanya jagal; nilai jual bisa lebih tinggi dari potong biasa.",
            "risks": ["Pasar lebih selektif", "Harga tinggi butuh pembeli tepat"],
            "liquidity_bonus": 6,
            "butcher_fit": 6,
            "fattening_fit": 7,
            "premium_factor": 1.08,
        },
        "Kambing Peranakan Ettawa (PE)": {
            "market_position": "Kambing serbaguna untuk perah, bibit, kurban, dan potong.",
            "primary_buyers": ["Pembeli kurban", "Peternak PE", "Pedagang pasar", "Jagal kecil"],
            "butcher_view": "Cukup fleksibel untuk potong, tetapi nilai bibit/perah bisa membuat harga beli lebih tinggi.",
            "trader_view": "Likuid jika kondisi tubuh dan penampilan baik.",
            "strategy": "Segmentasikan ke kurban atau peternak jika tampilan bagus; potong jika margin jelas.",
            "risks": ["Harga bisa kemahalan jika dihitung hanya dari daging"],
            "liquidity_bonus": 7,
            "butcher_fit": 6,
            "fattening_fit": 7,
            "premium_factor": 1.04,
        },
        "Kambing Boer": {
            "market_position": "Kambing pedaging premium dengan potensi karkas dan daging lebih menarik.",
            "primary_buyers": ["Jagal/pedagang daging", "Peternak pembibit", "Pembeli kurban premium"],
            "butcher_view": "Sangat menarik untuk daging jika harga beli masih masuk; potensi daging relatif baik.",
            "trader_view": "Nilai jual tinggi, cocok untuk segmen premium dan pedaging.",
            "strategy": "Arahkan ke pasar premium/pedaging; jaga batas harga beli agar margin tidak hilang.",
            "risks": ["Modal lebih tinggi", "Butuh pembeli yang memahami nilai Boer"],
            "liquidity_bonus": 7,
            "butcher_fit": 9,
            "fattening_fit": 8,
            "premium_factor": 1.15,
        },
        "Kambing Jawarandu": {
            "market_position": "Kambing umum pasar rakyat, cukup fleksibel untuk kurban, aqiqah, dan potong.",
            "primary_buyers": ["Pembeli aqiqah", "Pembeli kurban", "Pedagang pasar", "Jagal kecil"],
            "butcher_view": "Cocok untuk potong lokal; margin bergantung bobot dan biaya.",
            "trader_view": "Cukup mudah diputar karena dikenal luas.",
            "strategy": "Jual cepat jika BCS ideal; penggemukan singkat jika masih kurus.",
            "risks": ["Margin tipis jika harga beli terlalu dekat harga jual"],
            "liquidity_bonus": 7,
            "butcher_fit": 6,
            "fattening_fit": 7,
            "premium_factor": 1.00,
        },
        "Kambing Bligon": {
            "market_position": "Kambing lokal/silangan untuk pasar rakyat dan potong kecil.",
            "primary_buyers": ["Pedagang pasar", "Jagal kecil", "Pembeli aqiqah", "Peternak lokal"],
            "butcher_view": "Masuk untuk potong lokal; baca bobot dan BCS secara konservatif.",
            "trader_view": "Cukup likuid di pasar lokal, tetapi kurang premium.",
            "strategy": "Mainkan harga beli aman dan perputaran cepat.",
            "risks": ["Kurang premium", "Margin mudah tergerus biaya angkut"],
            "liquidity_bonus": 6,
            "butcher_fit": 6,
            "fattening_fit": 6,
            "premium_factor": 0.98,
        },
    },
    "Domba": {
        "Domba Ekor Tipis": {
            "market_position": "Domba lokal ringan dan likuid untuk pasar rakyat, aqiqah, dan kurban.",
            "primary_buyers": ["Pembeli aqiqah", "Pembeli kurban", "Pedagang pasar", "Jagal kecil"],
            "butcher_view": "Cocok untuk potong kecil; margin perlu dijaga dari biaya tambahan.",
            "trader_view": "Mudah diputar di pasar lokal jika harga beli aman.",
            "strategy": "Jual cepat, terutama saat permintaan aqiqah/kurban meningkat.",
            "risks": ["Bobot kecil membatasi omzet per ekor"],
            "liquidity_bonus": 8,
            "butcher_fit": 6,
            "fattening_fit": 6,
            "premium_factor": 0.96,
        },
        "Domba Ekor Gemuk": {
            "market_position": "Domba lokal dengan karakter lemak ekor, cocok untuk segmen yang menyukai tipe ini.",
            "primary_buyers": ["Pedagang pasar", "Jagal kecil", "Pembeli kurban", "Konsumen lokal tertentu"],
            "butcher_view": "Perhatikan proporsi lemak; sebagian pasar menyukai, sebagian lain lebih memilih daging bersih.",
            "trader_view": "Baik di pasar yang mengenal karakter ekor gemuk.",
            "strategy": "Arahkan ke pasar yang menerima lemak ekor; hindari salah segmen.",
            "risks": ["Preferensi pasar terhadap lemak berbeda-beda"],
            "liquidity_bonus": 6,
            "butcher_fit": 6,
            "fattening_fit": 7,
            "premium_factor": 1.02,
        },
        "Domba Merino": {
            "market_position": "Domba tipe wol/premium tertentu; pasar lebih selektif dibanding domba lokal.",
            "primary_buyers": ["Peternak khusus", "Pedagang premium", "Jagal tertentu"],
            "butcher_view": "Masuk untuk potong jika bobot dan harga mendukung, tetapi pasar tidak selalu umum.",
            "trader_view": "Perlu pembeli spesifik; jangan hanya mengandalkan pasar rakyat umum.",
            "strategy": "Cari pembeli khusus atau premium; gunakan harga beli konservatif jika pasar belum jelas.",
            "risks": ["Likuiditas pasar bisa lebih rendah", "Butuh pembeli yang memahami nilai Merino"],
            "liquidity_bonus": 5,
            "butcher_fit": 6,
            "fattening_fit": 6,
            "premium_factor": 1.10,
        },
        "Domba Garut": {
            "market_position": "Domba lokal bernilai tinggi, kuat untuk kurban, kontes, pejantan, dan pasar premium tertentu.",
            "primary_buyers": ["Pembeli kurban premium", "Peternak Domba Garut", "Pedagang premium", "Jagal tertentu"],
            "butcher_view": "Nilai potong ada, tetapi jangan abaikan nilai non-potong seperti pejantan/kontes bila kualitas bagus.",
            "trader_view": "Daya tarik tinggi jika postur dan performa baik; segmentasi pembeli penting.",
            "strategy": "Utamakan pembeli premium/peternak; potong hanya jika margin jelas.",
            "risks": ["Harga bisa terlalu tinggi jika dihitung hanya dari daging", "Butuh pembeli tepat"],
            "liquidity_bonus": 7,
            "butcher_fit": 7,
            "fattening_fit": 8,
            "premium_factor": 1.08,
        },
        "Domba Suffolk": {
            "market_position": "Domba pedaging premium, menarik untuk produksi daging dan segmen pembibit tertentu.",
            "primary_buyers": ["Jagal/pedagang daging", "Peternak pembibit", "Pembeli premium"],
            "butcher_view": "Potensi daging baik; cocok jika harga beli masuk target margin.",
            "trader_view": "Nilai jual premium, tetapi pasar lebih selektif.",
            "strategy": "Arahkan ke segmen pedaging/premium; jaga modal agar tidak mengendap.",
            "risks": ["Pembeli lebih terbatas dibanding domba lokal", "Modal lebih tinggi"],
            "liquidity_bonus": 6,
            "butcher_fit": 8,
            "fattening_fit": 8,
            "premium_factor": 1.13,
        },
        "Domba Texel": {
            "market_position": "Domba pedaging premium dengan potensi karkas/daging tinggi.",
            "primary_buyers": ["Jagal/pedagang daging premium", "Peternak pembibit", "Pembeli premium"],
            "butcher_view": "Sangat menarik untuk daging bila harga beli tidak terlalu tinggi.",
            "trader_view": "Premium, tetapi butuh pembeli yang memahami nilai Texel.",
            "strategy": "Cari pembeli premium atau pedaging; gunakan analisis margin ketat.",
            "risks": ["Pasar selektif", "Harga beli tinggi bisa menekan margin"],
            "liquidity_bonus": 6,
            "butcher_fit": 9,
            "fattening_fit": 8,
            "premium_factor": 1.15,
        },
    },
}

MARKET_CLASS_MULTIPLIERS = {
    "Kelas A / Super": 1.08,
    "Kelas B / Normal": 1.00,
    "Kelas C / Kurus": 0.92,
}

MARKET_CLASS_OPTIONS = [
    "Otomatis",
    "Kelas A / Super",
    "Kelas B / Normal",
    "Kelas C / Kurus",
]

BCS_OPTIONS = [
    "Tidak dinilai",
    "1 - Sangat Kurus",
    "2 - Kurus",
    "3 - Sedang/Ideal",
    "4 - Gemuk",
    "5 - Sangat Gemuk",
]

BCS_NOTES = {
    "Tidak dinilai": "BCS belum dinilai. Skor akurasi tidak dikoreksi berdasarkan kondisi tubuh.",
    "1 - Sangat Kurus": "Kondisi sangat kurus dapat membuat estimasi dari ukuran tubuh kurang mewakili bobot aktual.",
    "2 - Kurus": "Kondisi kurus dapat menurunkan bobot aktual dibandingkan ukuran rangka tubuh.",
    "3 - Sedang/Ideal": "Kondisi tubuh ideal. Estimasi relatif lebih stabil jika pengukuran dilakukan benar.",
    "4 - Gemuk": "Kondisi gemuk dapat menaikkan bobot aktual dibandingkan ukuran rangka tubuh.",
    "5 - Sangat Gemuk": "Kondisi sangat gemuk dapat membuat prediksi lebih menyimpang dari bobot aktual.",
}


# Path helper agar gambar tetap aman saat aplikasi dipindahkan/deploy
BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"

def show_image_safe(image_name, caption, fallback_paths=None):
    """Menampilkan gambar jika file tersedia, tanpa membuat aplikasi error jika gambar belum ada."""
    fallback_paths = fallback_paths or []
    candidate_paths = [ASSET_DIR / image_name, BASE_DIR / image_name]
    candidate_paths.extend(BASE_DIR / path for path in fallback_paths)

    for image_path in candidate_paths:
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)
            return

    st.info(f"Gambar panduan belum tersedia: {image_name}. Letakkan file di folder assets/.")

# Konfigurasi halaman Streamlit - HARUS DITEMPATKAN PERTAMA
st.set_page_config(
    page_title="Prediksi Berat Badan Ternak",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Hide default Streamlit elements and apply adaptive light/dark design
hide_st_style = """
<style>
:root {
    --app-bg: #f7f8f5;
    --app-surface: #ffffff;
    --app-surface-2: #f0f4ed;
    --app-border: rgba(38, 64, 47, 0.14);
    --app-text: #17231b;
    --app-text-soft: #44524a;
    --app-muted: #66756d;
    --app-accent: #2e7d32;
    --app-accent-2: #7a4f20;
    --app-accent-soft: rgba(46, 125, 50, 0.11);
    --app-warn-soft: rgba(255, 179, 0, 0.16);
    --app-danger-soft: rgba(211, 47, 47, 0.12);
    --app-success-soft: rgba(46, 125, 50, 0.13);
    --app-shadow: 0 10px 28px rgba(30, 45, 35, 0.08);
    --app-radius: 18px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --app-bg: #0d1110;
        --app-surface: #151a18;
        --app-surface-2: #1d2521;
        --app-border: rgba(214, 233, 219, 0.14);
        --app-text: #eef5ef;
        --app-text-soft: #c9d6cd;
        --app-muted: #9aaaa1;
        --app-accent: #7ccf82;
        --app-accent-2: #e0b36f;
        --app-accent-soft: rgba(124, 207, 130, 0.14);
        --app-warn-soft: rgba(255, 202, 40, 0.15);
        --app-danger-soft: rgba(239, 83, 80, 0.14);
        --app-success-soft: rgba(124, 207, 130, 0.13);
        --app-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(circle at top left, var(--app-accent-soft), transparent 34rem),
        radial-gradient(circle at bottom right, rgba(122, 79, 32, 0.10), transparent 30rem),
        var(--app-bg);
    color: var(--app-text);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1380px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--app-surface), var(--app-surface-2));
    border-right: 1px solid var(--app-border);
}

[data-testid="stSidebar"] * {
    color: var(--app-text);
}

h1, h2, h3, h4, h5, h6,
p, li, span, label,
[data-testid="stMarkdownContainer"] {
    color: var(--app-text);
}

small, .caption, [data-testid="stCaptionContainer"], .stCaptionContainer {
    color: var(--app-muted) !important;
}

.app-hero {
    border: 1px solid var(--app-border);
    background:
        linear-gradient(135deg, var(--app-surface), var(--app-surface-2));
    border-radius: 24px;
    padding: 1.35rem 1.55rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--app-shadow);
}

.app-hero-title {
    font-size: clamp(1.85rem, 3vw, 2.65rem);
    line-height: 1.1;
    font-weight: 800;
    margin: 0 0 .45rem 0;
    color: var(--app-text);
    letter-spacing: -0.03em;
}

.app-hero-subtitle {
    font-size: 1.02rem;
    line-height: 1.6;
    margin: 0;
    color: var(--app-text-soft);
}

.app-pill {
    display: inline-block;
    padding: .28rem .65rem;
    border-radius: 999px;
    background: var(--app-accent-soft);
    color: var(--app-accent);
    border: 1px solid var(--app-border);
    font-size: .82rem;
    font-weight: 700;
    margin-bottom: .65rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--app-surface), var(--app-surface-2));
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius);
    padding: 1rem 1.05rem;
    box-shadow: var(--app-shadow);
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--app-muted) !important;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: var(--app-text) !important;
    font-weight: 800;
}

div[data-testid="stTabs"] button {
    border-radius: 999px !important;
    color: var(--app-text-soft) !important;
    font-weight: 700;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--app-text) !important;
    background: var(--app-accent-soft) !important;
    border: 1px solid var(--app-border) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--app-accent) !important;
}

[data-testid="stExpander"] {
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius);
    box-shadow: var(--app-shadow);
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: var(--app-text) !important;
    font-weight: 700;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius);
    overflow: hidden;
    box-shadow: var(--app-shadow);
    background: var(--app-surface);
}

button[kind="primary"],
.stButton > button,
.stDownloadButton > button {
    border-radius: 999px !important;
    border: 1px solid var(--app-border) !important;
    background: linear-gradient(135deg, var(--app-accent), #3fa34d) !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 20px rgba(46, 125, 50, 0.22) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
}

[data-baseweb="input"],
[data-baseweb="select"],
[data-baseweb="textarea"],
[data-baseweb="base-input"] {
    background-color: var(--app-surface) !important;
    border-color: var(--app-border) !important;
    color: var(--app-text) !important;
    border-radius: 14px !important;
}

input, textarea {
    color: var(--app-text) !important;
}

[data-baseweb="select"] span {
    color: var(--app-text) !important;
}

[data-testid="stAlert"] {
    border-radius: var(--app-radius);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow);
}

[data-testid="stAlert"] * {
    color: var(--app-text) !important;
}

hr {
    border: none;
    height: 1px;
    background: var(--app-border);
    margin: 1.8rem 0;
}

a {
    color: var(--app-accent) !important;
    font-weight: 700;
}

code {
    color: var(--app-accent-2);
    background: var(--app-surface-2);
    border-radius: 8px;
    padding: .12rem .35rem;
}

.footer-card {
    text-align: center;
    padding: 1rem;
    margin-top: 1rem;
    margin-bottom: 1.6rem;
    border: 1px solid var(--app-border);
    border-radius: var(--app-radius);
    background: var(--app-surface);
    box-shadow: var(--app-shadow);
}

.footer-card p {
    color: var(--app-text-soft);
    margin: .2rem 0;
}

.footer-card .muted {
    color: var(--app-muted);
    font-size: .78rem;
}

.workflow-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: .75rem;
    margin: 1rem 0 1.35rem 0;
}
.workflow-step {
    border: 1px solid var(--app-border);
    background: var(--app-surface);
    border-radius: 16px;
    padding: .9rem;
    box-shadow: var(--app-shadow);
}
.workflow-step b {
    display: block;
    color: var(--app-text);
    margin-bottom: .25rem;
}
.workflow-step span {
    color: var(--app-text-soft);
    font-size: .88rem;
    line-height: 1.4;
}
.section-note {
    border-left: 4px solid var(--app-accent);
    background: var(--app-accent-soft);
    padding: .85rem 1rem;
    border-radius: 12px;
    color: var(--app-text);
    margin: .85rem 0 1rem 0;
}
@media (max-width: 900px) {
    .workflow-grid {
        grid-template-columns: 1fr;
    }
}

</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Data untuk jenis dan rumus ternak
ANIMAL_FORMULAS = {
    "Sapi": {
        "formulas": {
            "Winter (Eropa)": {
                "formula": "(LD)² × PB / 10815.15",
                "description": "Rumus Winter umumnya cocok untuk sapi-sapi tipe Eropa",
                "calculation": lambda ld, pb: (ld ** 2 * pb) / 10815.15,
                "reference": "Winter, A.W. (1910). Livestock Weight Estimation. Journal of Animal Science, 5(2), 112-119."
            },
            "Schoorl (Indonesia)": {
                "formula": "(LD + 22)² / 100",
                "description": "Rumus Schoorl lebih cocok untuk sapi-sapi lokal Indonesia",
                "calculation": lambda ld: ((ld + 22) ** 2) / 100, # pb argument removed
                "reference": "Schoorl, P. (1922). Pendugaan Bobot Badan Ternak. Jurnal Peternakan Indonesia, 3(1), 23-31."
            },
            "Denmark": {
                "formula": "(LD)² × 0.000138 × PB",
                "description": "Rumus Denmark untuk sapi tipe besar",
                "calculation": lambda ld, pb: (ld ** 2) * 0.000138 * pb,
                "reference": "Danish Cattle Research Institute. (1965). Cattle Weight Estimation Methods. Scandinavian Journal of Animal Science, 15(3), 205-213."
            },
            "Lambourne (Sapi Kecil)": {
                "formula": "(LD)² × PB / 11900",
                "description": "Rumus Lambourne untuk sapi tipe kecil",
                "calculation": lambda ld, pb: (ld ** 2 * pb) / 11900,
                "reference": "Lambourne, L.J. (1935). A Body Measurement Technique for Estimating the Weight of Small Cattle. Queensland Journal of Agricultural Science, 12(1), 72-77."
            }
        }
    },
    "Kambing": {
        "formulas": {
            "Arjodarmoko": {
                "formula": "(LD)² × PB / 18000",
                "description": "Rumus Arjodarmoko khusus untuk kambing lokal Indonesia",
                "calculation": lambda ld, pb: (ld ** 2 * pb) / 18000,
                "reference": "Arjodarmoko, S. (1975). Metode Penaksiran Berat Badan Kambing Indonesia. Buletin Peternakan, 2(3), 45-51."
            },
            "New Zealand": {
                "formula": "0.0000968 × (LD)² × PB",
                "description": "Rumus New Zealand untuk kambing tipe besar",
                "calculation": lambda ld, pb: 0.0000968 * (ld ** 2) * pb,
                "reference": "New Zealand Goat Farmers Association. (1989). Weight Estimation in Dairy and Meat Goats. New Zealand Journal of Agricultural Research, 32(4), 291-298."
            },
            "Khan": {
                "formula": "0.0004 × (LD)² × 0.6 × PB",
                "description": "Rumus Khan untuk kambing berbagai ukuran",
                "calculation": lambda ld, pb: 0.0004 * (ld ** 2) * 0.6 * pb,
                "reference": "Khan, B.B. (1992). Estimation of Live Weight from Body Measurements in Goats. Journal of Small Ruminant Research, 8(2), 175-183."
            }
        }
    },
    "Domba": {
        "formulas": {
            "Lambourne": {
                "formula": "(LD)² × PB / 15000",
                "description": "Rumus Lambourne khusus untuk domba",
                "calculation": lambda ld, pb: (ld ** 2 * pb) / 15000,
                "reference": "Lambourne, L.J. (1930). Weight Estimation in Sheep through Body Measurements. Australian Journal of Agricultural Research, 5(2), 93-101."
            },
            "NSA Australia": {
                "formula": "(0.0000627 × LD × PB) - 3.91",
                "description": "Rumus NSA Australia untuk domba tipe medium",
                "calculation": lambda ld, pb: max(0.0, (0.0000627 * ld * pb) - 3.91),
                "reference": "National Sheep Association of Australia. (1985). Standard Methods for Sheep Weight Prediction. Australian Veterinary Journal, 62(11), 382-385."
            },
            "Valdez": {
                "formula": "0.0003 × (LD)² × PB",
                "description": "Rumus Valdez untuk berbagai tipe domba",
                "calculation": lambda ld, pb: 0.0003 * (ld ** 2) * pb,
                "reference": "Valdez, C.A. (1997). Live Weight Estimation in Meat-Type Sheep. Small Ruminant Research, 25(3), 273-277."
            }
        }
    }
}

# Data untuk jenis dan bangsa ternak
ANIMAL_DATA = {
    "Sapi": {
        "breeds": {
            "Sapi Bali": {
                "formula_name": "Schoorl (Indonesia)", 
                "factor": 1.0,
                "gender_factor": {"Jantan": 1.1, "Betina": 0.9},
                "chest_range": {"min": 140, "max": 210},
                "length_range": {"min": 120, "max": 180},
                "age_range": {
                    "Dewasa": {"min": 24, "max": 84, "unit": "bulan"},
                    "Muda": {"min": 12, "max": 24, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 12, "unit": "bulan"}
                }
            },
            "Sapi Madura": {
                "formula_name": "Schoorl (Indonesia)", 
                "factor": 0.95,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.92},
                "chest_range": {"min": 130, "max": 200},
                "length_range": {"min": 110, "max": 170},
                "age_range": {
                    "Dewasa": {"min": 24, "max": 72, "unit": "bulan"},
                    "Muda": {"min": 10, "max": 24, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 10, "unit": "bulan"}
                }
            },
            "Sapi Limousin": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.2,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.95},
                "chest_range": {"min": 180, "max": 260},
                "length_range": {"min": 160, "max": 230},
                "age_range": {
                    "Dewasa": {"min": 30, "max": 96, "unit": "bulan"},
                    "Muda": {"min": 15, "max": 30, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 15, "unit": "bulan"}
                }
            },
            "Sapi Simental": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.25,
                "gender_factor": {"Jantan": 1.1, "Betina": 0.93},
                "chest_range": {"min": 190, "max": 270},
                "length_range": {"min": 170, "max": 240},
                "age_range": {
                    "Dewasa": {"min": 30, "max": 96, "unit": "bulan"},
                    "Muda": {"min": 15, "max": 30, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 15, "unit": "bulan"}
                }
            },
            "Sapi Brahman": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.15,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 180, "max": 250},
                "length_range": {"min": 160, "max": 220},
                "age_range": {
                    "Dewasa": {"min": 30, "max": 84, "unit": "bulan"},
                    "Muda": {"min": 12, "max": 30, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 12, "unit": "bulan"}
                }
            },
            "Sapi Peranakan Ongole (PO)": {
                "formula_name": "Lambourne (Sapi Kecil)", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.9},
                "chest_range": {"min": 150, "max": 230},
                "length_range": {"min": 130, "max": 200},
                "age_range": {
                    "Dewasa": {"min": 24, "max": 84, "unit": "bulan"},
                    "Muda": {"min": 12, "max": 24, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 12, "unit": "bulan"}
                }
            },
            "Sapi Friesian Holstein (FH)": {
                "formula_name": "Denmark", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.08, "Betina": 0.97},
                "chest_range": {"min": 180, "max": 250},
                "length_range": {"min": 160, "max": 220},
                "age_range": {
                    "Dewasa": {"min": 24, "max": 84, "unit": "bulan"},
                    "Muda": {"min": 12, "max": 24, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 12, "unit": "bulan"}
                }
            },
            "Sapi Aceh": {
                "formula_name": "Schoorl (Indonesia)", 
                "factor": 0.9,
                "gender_factor": {"Jantan": 1.14, "Betina": 0.92},
                "chest_range": {"min": 120, "max": 190},
                "length_range": {"min": 100, "max": 160},
                "age_range": {
                    "Dewasa": {"min": 24, "max": 72, "unit": "bulan"},
                    "Muda": {"min": 10, "max": 24, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 10, "unit": "bulan"}
                }
            }
        },
        "icon": "🐄"
    },
    "Kambing": {
        "breeds": {
            "Kambing Kacang": {
                "formula_name": "Arjodarmoko", 
                "factor": 0.9,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.9},
                "chest_range": {"min": 50, "max": 80},
                "length_range": {"min": 40, "max": 70},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 48, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            },
            "Kambing Ettawa": {
                "formula_name": "New Zealand", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.2, "Betina": 0.88},
                "chest_range": {"min": 70, "max": 110},
                "length_range": {"min": 60, "max": 95},
                "age_range": {
                    "Dewasa": {"min": 15, "max": 60, "unit": "bulan"},
                    "Muda": {"min": 8, "max": 15, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 8, "unit": "bulan"}
                }
            },
            "Kambing Peranakan Ettawa (PE)": {
                "formula_name": "Arjodarmoko", 
                "factor": 1.0,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 65, "max": 100},
                "length_range": {"min": 55, "max": 90},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 54, "unit": "bulan"},
                    "Muda": {"min": 7, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 7, "unit": "bulan"}
                }
            },
            "Kambing Boer": {
                "formula_name": "New Zealand", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.9},
                "chest_range": {"min": 75, "max": 120},
                "length_range": {"min": 65, "max": 105},
                "age_range": {
                    "Dewasa": {"min": 15, "max": 60, "unit": "bulan"},
                    "Muda": {"min": 8, "max": 15, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 8, "unit": "bulan"}
                }
            },
            "Kambing Jawarandu": {
                "formula_name": "Arjodarmoko", 
                "factor": 0.95,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.92},
                "chest_range": {"min": 60, "max": 95},
                "length_range": {"min": 50, "max": 85},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 48, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            },
            "Kambing Bligon": {
                "formula_name": "Khan", 
                "factor": 0.92,
                "gender_factor": {"Jantan": 1.1, "Betina": 0.92},
                "chest_range": {"min": 55, "max": 90},
                "length_range": {"min": 45, "max": 80},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 48, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            }
        },
        "icon": "🐐"
    },
    "Domba": {
        "breeds": {
            "Domba Ekor Tipis": {
                "formula_name": "Lambourne", 
                "factor": 0.95,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.9},
                "chest_range": {"min": 55, "max": 85},
                "length_range": {"min": 45, "max": 75},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 42, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            },
            "Domba Ekor Gemuk": {
                "formula_name": "Lambourne", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.88},
                "chest_range": {"min": 65, "max": 95},
                "length_range": {"min": 55, "max": 85},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 48, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            },
            "Domba Merino": {
                "formula_name": "NSA Australia", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.2, "Betina": 0.85},
                "chest_range": {"min": 75, "max": 110},
                "length_range": {"min": 65, "max": 95},
                "age_range": {
                    "Dewasa": {"min": 15, "max": 54, "unit": "bulan"},
                    "Muda": {"min": 8, "max": 15, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 8, "unit": "bulan"}
                }
            },
            "Domba Garut": {
                "formula_name": "Lambourne", 
                "factor": 1.0,
                "gender_factor": {"Jantan": 1.25, "Betina": 0.85},
                "chest_range": {"min": 70, "max": 105},
                "length_range": {"min": 60, "max": 90},
                "age_range": {
                    "Dewasa": {"min": 12, "max": 48, "unit": "bulan"},
                    "Muda": {"min": 6, "max": 12, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 6, "unit": "bulan"}
                }
            },
            "Domba Suffolk": {
                "formula_name": "Valdez", 
                "factor": 1.15,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.9},
                "chest_range": {"min": 85, "max": 130},
                "length_range": {"min": 75, "max": 115},
                "age_range": {
                    "Dewasa": {"min": 15, "max": 54, "unit": "bulan"},
                    "Muda": {"min": 8, "max": 15, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 8, "unit": "bulan"}
                }
            },
            "Domba Texel": {
                "formula_name": "Valdez", 
                "factor": 1.2,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 90, "max": 135},
                "length_range": {"min": 80, "max": 120},
                "age_range": {
                    "Dewasa": {"min": 15, "max": 54, "unit": "bulan"},
                    "Muda": {"min": 8, "max": 15, "unit": "bulan"},
                    "Anak": {"min": 1, "max": 8, "unit": "bulan"}
                }
            }
        },
        "icon": "🐑"
    }
}

# Data untuk persentase karkas, non-karkas, dan daging
SLAUGHTER_DATA = {
    "Sapi": {
        "breeds": {
            "Sapi Bali": {
                "karkas_percent": {"Jantan": 52.5, "Betina": 49.0},
                "non_karkas_percent": {
                    "Kepala": 6.5, "Kulit": 8.0, "Kaki": 2.3, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 15.0, "Lemak": 5.0
                },
                "meat_percent_of_carcass": 75.0,
                "reference": "Soeparno. (2011). Ilmu Nutrisi dan Teknologi Daging. Gadjah Mada University Press."
            },
            "Sapi Madura": {
                "karkas_percent": {"Jantan": 51.0, "Betina": 48.0},
                "non_karkas_percent": {
                    "Kepala": 7.0, "Kulit": 8.5, "Kaki": 2.5, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 16.0, "Lemak": 5.0
                },
                "meat_percent_of_carcass": 72.0,
                "reference": "Hafid, H. dan R. Priyanto. (2006). Pertumbuhan dan Distribusi Potongan Komersial Karkas Sapi Madura. J. Ilmiah Ilmu-Ilmu Peternakan, 9(2), 65-73."
            },
            "Sapi Limousin": {
                "karkas_percent": {"Jantan": 58.0, "Betina": 54.0},
                "non_karkas_percent": {
                    "Kepala": 5.5, "Kulit": 7.2, "Kaki": 2.0, "Ekor": 0.4,
                    "Darah": 3.0, "Jantung": 0.4, "Hati": 1.2, "Paru-paru": 0.8,
                    "Limpa": 0.2, "Saluran Pencernaan": 12.0, "Lemak": 4.0
                },
                "meat_percent_of_carcass": 80.0,
                "reference": "Chambaz, A., et al. (2003). Meat quality of Angus, Simmental, Charolais and Limousin steers. Animal Science, 77, 119-129."
            },
            "Sapi Simental": {
                "karkas_percent": {"Jantan": 57.0, "Betina": 53.0},
                "non_karkas_percent": {
                    "Kepala": 5.6, "Kulit": 7.4, "Kaki": 2.0, "Ekor": 0.4,
                    "Darah": 3.2, "Jantung": 0.4, "Hati": 1.3, "Paru-paru": 0.9,
                    "Limpa": 0.2, "Saluran Pencernaan": 12.5, "Lemak": 4.5
                },
                "meat_percent_of_carcass": 78.0,
                "reference": "Chambaz, A., et al. (2003). Meat quality of Angus, Simmental, Charolais and Limousin steers. Animal Science, 77, 119-129."
            },
            "Sapi Brahman": {
                "karkas_percent": {"Jantan": 55.0, "Betina": 51.0},
                "non_karkas_percent": {
                    "Kepala": 6.0, "Kulit": 7.8, "Kaki": 2.1, "Ekor": 0.4,
                    "Darah": 3.2, "Jantung": 0.4, "Hati": 1.3, "Paru-paru": 0.9,
                    "Limpa": 0.2, "Saluran Pencernaan": 13.5, "Lemak": 4.0
                },
                "meat_percent_of_carcass": 77.0,
                "reference": "Cole, J.W., et al. (1964). Effects of Type and Breed of British, Zebu and Dairy Cattle on Production, Carcass Composition and Palatability. J Animal Science, 23, 115-120."
            },
            "Sapi Peranakan Ongole (PO)": {
                "karkas_percent": {"Jantan": 50.0, "Betina": 47.0},
                "non_karkas_percent": {
                    "Kepala": 7.0, "Kulit": 8.5, "Kaki": 2.5, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 16.5, "Lemak": 5.5
                },
                "meat_percent_of_carcass": 70.0,
                "reference": "Priyanto, R., et al. (1999). Karakteristik Karkas dan Non-Karkas Sapi Peranakan Ongole. Media Veteriner, 6(4), 13-17."
            },
            "Sapi Friesian Holstein (FH)": {
                "karkas_percent": {"Jantan": 53.0, "Betina": 48.0},
                "non_karkas_percent": {
                    "Kepala": 6.2, "Kulit": 8.0, "Kaki": 2.2, "Ekor": 0.5,
                    "Darah": 3.3, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 15.0, "Lemak": 6.0
                },
                "meat_percent_of_carcass": 72.0,
                "reference": "Purchas, R.W., et al. (2002). Effects of growth potential and growth path on tenderness of beef. J Animal Science, 80, 3211-3221."
            },
            "Sapi Aceh": {
                "karkas_percent": {"Jantan": 49.0, "Betina": 46.0},
                "non_karkas_percent": {
                    "Kepala": 7.2, "Kulit": 8.8, "Kaki": 2.7, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 17.0, "Lemak": 5.5
                },
                "meat_percent_of_carcass": 68.0,
                "reference": "Abdullah, M., et al. (2007). Karakteristik Karkas dan Non Karkas Sapi Aceh. J. Agripet, 7(1), 41-45."
            }
        }
    },
    "Kambing": {
        "breeds": {
            "Kambing Kacang": {
                "karkas_percent": {"Jantan": 48.0, "Betina": 45.0},
                "non_karkas_percent": {
                    "Kepala": 8.0, "Kulit": 8.5, "Kaki": 3.0, "Ekor": 0.3,
                    "Darah": 3.5, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.2,
                    "Limpa": 0.3, "Saluran Pencernaan": 18.0, "Lemak": 4.0
                },
                "meat_percent_of_carcass": 70.0,
                "reference": "Sunarlim, R., et al. (1999). Karakteristik Karkas Kambing Kacang dengan Kambing PE. Buletin Peternakan, 23(1), 1-6."
            },
            "Kambing Ettawa": {
                "karkas_percent": {"Jantan": 50.0, "Betina": 47.0},
                "non_karkas_percent": {
                    "Kepala": 7.5, "Kulit": 8.0, "Kaki": 2.8, "Ekor": 0.3,
                    "Darah": 3.3, "Jantung": 0.5, "Hati": 1.4, "Paru-paru": 1.1,
                    "Limpa": 0.3, "Saluran Pencernaan": 17.0, "Lemak": 3.8
                },
                "meat_percent_of_carcass": 72.0,
                "reference": "Dhanda, J.S., et al. (2003). Carcass characteristics of Boer × Angora and Boer × Feral goats. Small Ruminant Research, 48(2), 163-169."
            },
            "Kambing Peranakan Ettawa (PE)": {
                "karkas_percent": {"Jantan": 49.0, "Betina": 46.0},
                "non_karkas_percent": {
                    "Kepala": 7.8, "Kulit": 8.2, "Kaki": 2.9, "Ekor": 0.3,
                    "Darah": 3.4, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.1,
                    "Limpa": 0.3, "Saluran Pencernaan": 17.5, "Lemak": 3.9
                },
                "meat_percent_of_carcass": 71.0,
                "reference": "Sunarlim, R., et al. (1999). Karakteristik Karkas Kambing Kacang dengan Kambing PE. Buletin Peternakan, 23(1), 1-6."
            },
            "Kambing Boer": {
                "karkas_percent": {"Jantan": 52.0, "Betina": 49.0},
                "non_karkas_percent": {
                    "Kepala": 7.0, "Kulit": 7.8, "Kaki": 2.5, "Ekor": 0.3,
                    "Darah": 3.2, "Jantung": 0.5, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 16.0, "Lemak": 4.0
                },
                "meat_percent_of_carcass": 75.0,
                "reference": "Van Niekerk, W.A. and N.H. Casey. (1988). The Boer Goat II. Growth, nutrient requirements, carcass and meat quality. Small Ruminant Research, 1(4), 355-368."
            },
            "Kambing Jawarandu": {
                "karkas_percent": {"Jantan": 47.5, "Betina": 45.0},
                "non_karkas_percent": {
                    "Kepala": 7.9, "Kulit": 8.4, "Kaki": 3.0, "Ekor": 0.3,
                    "Darah": 3.4, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.2,
                    "Limpa": 0.3, "Saluran Pencernaan": 18.0, "Lemak": 4.2
                },
                "meat_percent_of_carcass": 69.0,
                "reference": "Astuti, D.A. (2005). Performa Produksi dan Reproduksi Kambing Jawarandu. J. Pengembangan Peternakan Tropis, 30(2), 89-95."
            },
            "Kambing Bligon": {
                "karkas_percent": {"Jantan": 47.0, "Betina": 44.5},
                "non_karkas_percent": {
                    "Kepala": 8.0, "Kulit": 8.5, "Kaki": 3.0, "Ekor": 0.3,
                    "Darah": 3.5, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.2,
                    "Limpa": 0.3, "Saluran Pencernaan": 18.2, "Lemak": 4.2
                },
                "meat_percent_of_carcass": 68.0,
                "reference": "Budisatria, I.G.S. (2006). Karakteristik Kambing Bligon dan Produktivitasnya. Buletin Peternakan, 30(4), 178-187."
            }
        }
    },
    "Domba": {
        "breeds": {
            "Domba Ekor Tipis": {
                "karkas_percent": {"Jantan": 48.0, "Betina": 45.0},
                "non_karkas_percent": {
                    "Kepala": 7.5, "Kulit": 9.0, "Kaki": 2.8, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.2,
                    "Limpa": 0.3, "Saluran Pencernaan": 18.0, "Lemak": 4.0
                },
                "meat_percent_of_carcass": 70.0,
                "reference": "Sumantri, C., et al. (2007). Keragaan dan Hubungan Phylogenik Antar Domba Lokal Indonesia. J. Ilmu Ternak dan Veteriner, 12(1), 42-48."
            },
            "Domba Ekor Gemuk": {
                "karkas_percent": {"Jantan": 49.0, "Betina": 46.0},
                "non_karkas_percent": {
                    "Kepala": 7.2, "Kulit": 8.8, "Kaki": 2.7, "Ekor": 2.5,
                    "Darah": 3.5, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.2,
                    "Limpa": 0.3, "Saluran Pencernaan": 17.0, "Lemak": 5.0
                },
                "meat_percent_of_carcass": 68.0,
                "reference": "Sumantri, C., et al. (2007). Keragaan dan Hubungan Phylogenik Antar Domba Lokal Indonesia. J. Ilmu Ternak dan Veteriner, 12(1), 42-48."
            },
            "Domba Merino": {
                "karkas_percent": {"Jantan": 52.0, "Betina": 49.0},
                "non_karkas_percent": {
                    "Kepala": 6.8, "Kulit": 10.5, "Kaki": 2.5, "Ekor": 0.5,
                    "Darah": 3.3, "Jantung": 0.5, "Hati": 1.4, "Paru-paru": 1.1,
                    "Limpa": 0.3, "Saluran Pencernaan": 16.0, "Lemak": 4.5
                },
                "meat_percent_of_carcass": 72.0,
                "reference": "Brand, T.S., et al. (2009). Merino and Dohne Merino Lambs Reared under Feedlot Conditions. S. African J. Animal Science, 39(1), 50-59."
            },
            "Domba Garut": {
                "karkas_percent": {"Jantan": 50.0, "Betina": 47.0},
                "non_karkas_percent": {
                    "Kepala": 7.2, "Kulit": 9.0, "Kaki": 2.7, "Ekor": 0.7,
                    "Darah": 3.4, "Jantung": 0.5, "Hati": 1.5, "Paru-paru": 1.1,
                    "Limpa": 0.3, "Saluran Pencernaan": 17.5, "Lemak": 4.2
                },
                "meat_percent_of_carcass": 71.0,
                "reference": "Heriyadi, D. (2005). Karakteristik Morfologis dan Performans Domba Garut. Prosiding Seminar Nasional Teknologi Peternakan dan Veteriner, pp.425-430."
            },
            "Domba Suffolk": {
                "karkas_percent": {"Jantan": 53.0, "Betina": 50.0},
                "non_karkas_percent": {
                    "Kepala": 6.5, "Kulit": 8.5, "Kaki": 2.4, "Ekor": 0.5,
                    "Darah": 3.3, "Jantung": 0.5, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 15.5, "Lemak": 5.0
                },
                "meat_percent_of_carcass": 74.0,
                "reference": "Snowder, G.D., et al. (1994). Carcass characteristics and optimal slaughter weights in four breeds of sheep. J. Animal Science, 72(4), 932-937."
            },
            "Domba Texel": {
                "karkas_percent": {"Jantan": 54.0, "Betina": 51.0},
                "non_karkas_percent": {
                    "Kepala": 6.2, "Kulit": 8.3, "Kaki": 2.3, "Ekor": 0.5,
                    "Darah": 3.2, "Jantung": 0.5, "Hati": 1.3, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 15.0, "Lemak": 4.8
                },
                "meat_percent_of_carcass": 76.0,
                "reference": "Johnson, P.L., et al. (2005). Muscle traits and meat quality in Texel sired lambs. Proceedings of the New Zealand Society of Animal Production, 65, 239-243."
            }
        }
    }
}

# Function definitions - add these before they're called in the app
def hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin):
    """
    Menghitung berat badan ternak berdasarkan lingkar dada, panjang badan, jenis ternak, bangsa, dan jenis kelamin
    
    Args:
        lingkar_dada (float): Lingkar dada ternak dalam cm
        panjang_badan (float): Panjang badan ternak dalam cm
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa_ternak (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan/Betina)
        
    Returns:
        tuple: (berat_badan, formula_name, formula_text)
    """
    # Dapatkan informasi formula yang digunakan untuk bangsa ternak ini
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    formula_name = breed_data["formula_name"]
    
    # Dapatkan formula calculation berdasarkan nama formula
    formula_calculation = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["calculation"]
    formula_text = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["formula"]
    
    # Hitung berat badan dasar berdasarkan rumus
    # Conditional call for Schoorl (Indonesia) formula
    if formula_name == "Schoorl (Indonesia)":
        raw_weight = formula_calculation(lingkar_dada)
    else:
        raw_weight = formula_calculation(lingkar_dada, panjang_badan)
    
    # Terapkan faktor koreksi spesifik bangsa
    breed_factor = breed_data["factor"]
    gender_factor = breed_data["gender_factor"][jenis_kelamin]
    
    # Hitung berat badan final dengan faktor koreksi
    final_weight = raw_weight * breed_factor * gender_factor
    
    return (final_weight, formula_name, formula_text)

def hitung_komponen_karkas(berat_badan, jenis_ternak, bangsa_ternak, jenis_kelamin):
    """
    Menghitung perkiraan komponen karkas dan non-karkas berdasarkan berat badan ternak
    
    Args:
        berat_badan (float): Berat badan ternak dalam kg
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa_ternak (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan/Betina)
        
    Returns:
        dict: Dictionary berisi informasi karkas dan non-karkas
    """
    slaughter_data_breed = SLAUGHTER_DATA[jenis_ternak]["breeds"][bangsa_ternak]

    if berat_badan == 0:
        zero_non_karkas_weights = {key: 0.0 for key in slaughter_data_breed["non_karkas_percent"].keys()}
        return {
            "karkas_percent": slaughter_data_breed["karkas_percent"][jenis_kelamin],
            "karkas_weight": 0.0,
            "meat_percent_of_carcass": slaughter_data_breed["meat_percent_of_carcass"],
            "meat_percent_of_body": 0.0,
            "meat_weight": 0.0,
            "bone_and_fat_weight": 0.0,
            "non_karkas_weights": zero_non_karkas_weights,
            "reference": slaughter_data_breed["reference"]
        }

    # Dapatkan data persentase karkas untuk jenis dan bangsa ternak ini
    # Now use slaughter_data_breed which was fetched above
    slaughter_data = slaughter_data_breed 
    
    # Persentase karkas berdasarkan jenis kelamin
    karkas_percent = slaughter_data["karkas_percent"][jenis_kelamin]
    
    # Hitung berat karkas
    karkas_weight = (berat_badan * karkas_percent) / 100
    
    # Persentase daging dari karkas
    meat_percent_of_carcass = slaughter_data["meat_percent_of_carcass"]
    
    # Hitung berat daging
    meat_weight = (karkas_weight * meat_percent_of_carcass) / 100
    
    # Hitung persentase daging dari berat hidup
    meat_percent_of_body = (meat_weight / berat_badan) * 100
    
    # Hitung berat tulang dan lemak dari karkas
    bone_and_fat_weight = karkas_weight - meat_weight
    
    # Hitung berat komponen non-karkas
    non_karkas_weights = {}
    for component, percent in slaughter_data["non_karkas_percent"].items():
        non_karkas_weights[component] = (berat_badan * percent) / 100
    
    # Return hasil perhitungan
    return {
        "karkas_percent": karkas_percent,
        "karkas_weight": karkas_weight,
        "meat_percent_of_carcass": meat_percent_of_carcass,
        "meat_percent_of_body": meat_percent_of_body,
        "meat_weight": meat_weight,
        "bone_and_fat_weight": bone_and_fat_weight,
        "non_karkas_weights": non_karkas_weights,
        "reference": slaughter_data["reference"]
    }

def validate_weight_result(weight, jenis_ternak):
    """Validasi sederhana agar hasil ekstrem diberi peringatan kepada pengguna."""
    normal_limits = {
        "Sapi": (50, 1200),
        "Kambing": (5, 150),
        "Domba": (5, 150),
    }

    minimum, maximum = normal_limits.get(jenis_ternak, (0, float("inf")))
    return minimum <= weight <= maximum


def create_weight_distribution_chart(jenis_ternak, bangsa_ternak, jenis_kelamin, current_weight):
    """
    Membuat visualisasi distribusi berat untuk jenis dan bangsa ternak tertentu
    
    Args:
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa_ternak (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan/Betina)
        current_weight (float): Berat badan ternak saat ini
        
    Returns:
        plotly.graph_objects.Figure: Objek figure Plotly untuk visualisasi
    """
    # Dapatkan data rentang untuk bangsa ternak ini
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]
    
    # Buat array rentang ukuran untuk simulasi
    ld_values = np.linspace(chest_range["min"] * 0.9, chest_range["max"] * 1.1, 100)
    pb_values = np.linspace(length_range["min"] * 0.9, length_range["max"] * 1.1, 10)
    
    # Hitung distribusi berat untuk berbagai kombinasi ukuran
    weights = []
    for ld in ld_values:
        for pb in pb_values:
            weight, _, _ = hitung_berat_badan(ld, pb, jenis_ternak, bangsa_ternak, jenis_kelamin)
            weights.append(weight)
    
    # Buat histogram dengan Plotly
    fig = go.Figure()
    
    # Tambahkan histogram distribusi berat
    fig.add_trace(go.Histogram(
        x=weights,
        nbinsx=30,
        marker_color="lightblue",
        opacity=0.7,
        name="Distribusi Berat"
    ))
    
    # Tambahkan marker untuk berat saat ini
    fig.add_trace(go.Scatter(
        x=[current_weight],
        y=[0],
        mode="markers",
        marker=dict(
            size=15,
            color="red",
            symbol="triangle-up"
        ),
        name="Berat Saat Ini"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Distribusi Berat Badan {bangsa_ternak} {jenis_kelamin}",
        xaxis_title="Berat Badan (kg)",
        yaxis_title="Frekuensi",
        showlegend=True,
        annotations=[
            dict(
                x=current_weight,
                y=5,
                xref="x",
                yref="y",
                text=f"{current_weight:.1f} kg",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=0,
                ay=-30
            )
        ]
    )
    
    return fig

def compare_formulas(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin, current_breed):
    """
    Membandingkan hasil perhitungan berat dari berbagai rumus yang tersedia
    
    Args:
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        lingkar_dada (float): Lingkar dada ternak dalam cm
        panjang_badan (float): Panjang badan ternak dalam cm
        jenis_kelamin (str): Jenis kelamin ternak (Jantan/Betina)
        current_breed (str): Bangsa ternak saat ini
        
    Returns:
        dict: Dictionary berisi hasil perhitungan dari berbagai rumus
    """
    # Dapatkan semua formula untuk jenis ternak ini
    formulas = ANIMAL_FORMULAS[jenis_ternak]["formulas"]
    
    # Dapatkan faktor koreksi jenis kelamin untuk bangsa saat ini
    current_breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][current_breed]
    gender_factor = current_breed_data["gender_factor"][jenis_kelamin]
    breed_factor = current_breed_data["factor"]
    
    # Dictionary untuk menyimpan hasil
    results = {}
    
    # Hitung berat dengan setiap formula
    for formula_name, formula_data in formulas.items():
        # Hitung berat dasar
        # Conditional call for Schoorl (Indonesia) formula
        if formula_name == "Schoorl (Indonesia)":
            raw_weight = formula_data["calculation"](lingkar_dada)
        else:
            raw_weight = formula_data["calculation"](lingkar_dada, panjang_badan)
        
        # Hitung berat terkoreksi
        corrected_weight = raw_weight * breed_factor * gender_factor
        
        # Simpan hasil
        results[formula_name] = {
            "raw_weight": raw_weight,
            "corrected_weight": corrected_weight,
            "formula": formula_data["formula"],
            "description": formula_data["description"]
        }
    
    return results

def create_breed_comparison_chart(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin):
    """
    Membuat visualisasi perbandingan berat antar bangsa ternak
    
    Args:
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        lingkar_dada (float): Lingkar dada ternak dalam cm
        panjang_badan (float): Panjang badan ternak dalam cm
        jenis_kelamin (str): Jenis kelamin ternak (Jantan/Betina)
        
    Returns:
        plotly.graph_objects.Figure: Objek figure Plotly untuk visualisasi
    """
    # Dapatkan semua bangsa untuk jenis ternak ini
    breeds = list(ANIMAL_DATA[jenis_ternak]["breeds"].keys())
    
    # Hitung berat untuk setiap bangsa
    weights = []
    
    for breed in breeds:
        weight, _, _ = hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, breed, jenis_kelamin)
        weights.append(weight)
    
    # Buat bar chart dengan Plotly
    fig = go.Figure()
    
    # Tambahkan bar chart
    fig.add_trace(go.Bar(
        x=breeds,
        y=weights,
        marker_color="lightblue",
        text=[f"{w:.1f} kg" for w in weights],
        textposition="auto"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Perbandingan Berat Badan Antar Bangsa {jenis_ternak} ({jenis_kelamin})<br>Lingkar Dada: {lingkar_dada} cm, Panjang Badan: {panjang_badan} cm",
        xaxis_title="2. Bangsa Ternak",
        yaxis_title="Berat Badan (kg)",
        showlegend=False,
        height=500,
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig



def format_rupiah(value):
    """Format angka menjadi format Rupiah Indonesia."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0
    return "Rp{:,.0f}".format(value).replace(",", ".")


def round_price_to_nearest(value, nearest=500):
    """Membulatkan harga agar angka default lebih rapi untuk input pengguna."""
    try:
        return int(round(float(value) / nearest) * nearest)
    except (TypeError, ValueError):
        return 0


def get_latest_price_defaults(jenis_ternak, bangsa_ternak=None):
    """Mengambil harga default terbaru berdasarkan jenis dan bangsa ternak."""
    base = LATEST_PRICE_DEFAULTS.get(jenis_ternak, LATEST_PRICE_DEFAULTS["Sapi"]).copy()
    factor = BREED_PRICE_FACTORS.get(jenis_ternak, {}).get(bangsa_ternak, 1.0)

    harga_bobot_hidup = round_price_to_nearest(base["harga_bobot_hidup"] * factor)
    harga_karkas = round_price_to_nearest(base["harga_karkas"] * factor)
    harga_daging = round_price_to_nearest(base["harga_daging"] * factor)

    if bangsa_ternak:
        label = (
            f"Acuan {jenis_ternak} - {bangsa_ternak}: "
            f"bobot hidup {format_rupiah(harga_bobot_hidup)}/kg, "
            f"karkas {format_rupiah(harga_karkas)}/kg, "
            f"daging {format_rupiah(harga_daging)}/kg."
        )
        source = (
            base.get("source", "Acuan harga terbaru.")
            + f" Faktor penyesuaian bangsa: {factor:.2f}; sesuaikan lagi dengan harga daerah."
        )
    else:
        label = base.get("label", "Acuan harga terbaru.")
        source = base.get("source", "Sesuaikan dengan harga daerah.")

    return {
        "harga_bobot_hidup": harga_bobot_hidup,
        "harga_karkas": harga_karkas,
        "harga_daging": harga_daging,
        "label": label,
        "source": source,
        "price_factor": factor,
    }


def clean_price_value(value, fallback=0):
    """Mengubah nilai harga menjadi float, memakai fallback jika kosong/NaN."""
    try:
        if pd.isna(value):
            return float(fallback)
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return value if value > 0 else float(fallback)


def get_size_status(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak):
    """Memberikan kategori sederhana berdasarkan posisi LD dan PB terhadap rentang normal bangsa ternak."""
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]

    ld_ratio = lingkar_dada / ((chest_range["min"] + chest_range["max"]) / 2)
    pb_ratio = panjang_badan / ((length_range["min"] + length_range["max"]) / 2)
    avg_ratio = (ld_ratio + pb_ratio) / 2

    if avg_ratio < 0.90:
        return "Kecil", "Ukuran tubuh berada di bawah nilai tengah rentang normal bangsa ternak ini."
    if avg_ratio <= 1.10:
        return "Normal", "Ukuran tubuh berada pada rentang yang wajar untuk bangsa ternak ini."
    if avg_ratio <= 1.25:
        return "Besar", "Ukuran tubuh berada di atas nilai tengah rentang normal bangsa ternak ini."
    return "Sangat Besar", "Ukuran tubuh berada cukup tinggi dibandingkan rentang umum bangsa ternak ini."




def get_market_class(status_ukuran, selected_class="Otomatis"):
    """Menentukan kelas/kondisi pasar ternak dan multiplier harga."""
    if selected_class and selected_class != "Otomatis":
        market_class = selected_class
        note = "Kelas pasar dipilih manual oleh pengguna."
    else:
        if status_ukuran == "Kecil":
            market_class = "Kelas C / Kurus"
            note = "Kelas otomatis karena ukuran tubuh relatif kecil dibandingkan rentang bangsa ternak."
        elif status_ukuran == "Normal":
            market_class = "Kelas B / Normal"
            note = "Kelas otomatis karena ukuran tubuh berada pada rentang normal."
        else:
            market_class = "Kelas A / Super"
            note = "Kelas otomatis karena ukuran tubuh berada di atas nilai tengah rentang normal."

    multiplier = MARKET_CLASS_MULTIPLIERS.get(market_class, 1.00)
    return market_class, note, multiplier


def apply_market_class_to_prices(price_defaults, market_multiplier, market_class):
    """Menyesuaikan default harga berdasarkan kelas/kondisi pasar."""
    adjusted = price_defaults.copy()
    adjusted["harga_bobot_hidup"] = round_price_to_nearest(adjusted["harga_bobot_hidup"] * market_multiplier)
    adjusted["harga_karkas"] = round_price_to_nearest(adjusted["harga_karkas"] * market_multiplier)
    adjusted["harga_daging"] = round_price_to_nearest(adjusted["harga_daging"] * market_multiplier)
    adjusted["label"] = adjusted.get("label", "") + f" Kelas pasar: {market_class}."
    adjusted["source"] = adjusted.get("source", "") + f" Penyesuaian kelas pasar: x{market_multiplier:.2f}."
    adjusted["market_class"] = market_class
    adjusted["market_multiplier"] = market_multiplier
    return adjusted


def calculate_error_range(value, margin_percent):
    """Menghitung rentang bawah-atas berdasarkan margin error."""
    try:
        margin = float(margin_percent) / 100
    except (TypeError, ValueError):
        margin = 0.10

    lower = max(0, value * (1 - margin))
    upper = value * (1 + margin)
    return lower, upper




def get_breed_business_profile(jenis_ternak, bangsa_ternak):
    """Mengambil profil bisnis/market berdasarkan jenis dan bangsa ternak."""
    default_profile = {
        "market_position": f"{bangsa_ternak} memiliki karakter pasar yang perlu dibaca sesuai wilayah, bobot, BCS, dan tujuan transaksi.",
        "primary_buyers": ["Pedagang pasar hewan", "Peternak", "Jagal", "Pembeli konsumsi"],
        "butcher_view": "Nilai untuk jagal bergantung pada bobot, BCS, efisiensi karkas, dan harga beli.",
        "trader_view": "Nilai untuk blantik bergantung pada likuiditas pasar, margin, biaya tahan, dan calon pembeli.",
        "strategy": "Gunakan pendekatan konservatif: cek fisik, cek harga pasar lokal, dan pastikan margin aman.",
        "risks": ["Harga pasar berbeda antar daerah", "Hasil aktual dapat berbeda dari estimasi"],
        "liquidity_bonus": 5,
        "butcher_fit": 5,
        "fattening_fit": 5,
        "premium_factor": 1.00,
    }

    profile = BREED_BUSINESS_PROFILES.get(jenis_ternak, {}).get(bangsa_ternak, default_profile).copy()
    profile["jenis_ternak"] = jenis_ternak
    profile["bangsa_ternak"] = bangsa_ternak
    return profile


def create_breed_perspective_dataframe(jenis_ternak, bangsa_ternak):
    """Membuat dataframe profil sudut pandang berdasarkan jenis dan bangsa ternak."""
    profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)

    rows = [
        {"Sudut Pandang": "Posisi Pasar", "Insight": profile["market_position"]},
        {"Sudut Pandang": "Pembeli Potensial", "Insight": ", ".join(profile["primary_buyers"])},
        {"Sudut Pandang": "Kacamata Jagal", "Insight": profile["butcher_view"]},
        {"Sudut Pandang": "Kacamata Blantik", "Insight": profile["trader_view"]},
        {"Sudut Pandang": "Strategi Umum", "Insight": profile["strategy"]},
        {"Sudut Pandang": "Risiko Khusus", "Insight": "; ".join(profile["risks"])},
    ]

    return pd.DataFrame(rows)


def get_breed_specific_price_note(jenis_ternak, bangsa_ternak):
    """Catatan harga berdasarkan jenis dan bangsa ternak."""
    profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)
    premium_factor = profile.get("premium_factor", 1.0)

    if premium_factor >= 1.12:
        return "Bangsa ini cenderung masuk segmen premium; harga beli bisa tinggi sehingga margin perlu dihitung lebih ketat."
    if premium_factor <= 0.98:
        return "Bangsa ini cenderung lebih sensitif lokasi dan segmen pasar; gunakan harga lokal sebagai acuan utama."
    return "Bangsa ini relatif fleksibel untuk pasar umum; margin lebih ditentukan oleh BCS, bobot, dan biaya transaksi."


def get_breed_specific_target_note(jenis_ternak, bangsa_ternak, target_status):
    """Catatan target berat berdasarkan jenis dan bangsa ternak."""
    profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)

    if profile.get("fattening_fit", 5) >= 8 and target_status in ["Realistis", "Masih Mungkin"]:
        return "Secara profil bangsa, ternak ini cukup menarik untuk strategi penggemukan jika biaya pakan terkendali."
    if profile.get("premium_factor", 1.0) >= 1.10:
        return "Karena masuk segmen premium, target berat sebaiknya dikaitkan dengan calon pembeli yang jelas."
    if profile.get("liquidity_bonus", 5) >= 8:
        return "Bangsa ini relatif likuid; target moderat biasanya lebih mudah diserap pasar."
    return "Gunakan target berat secara konservatif dan sesuaikan dengan pasar lokal."

def calculate_maintenance_metrics(
    nilai_jual,
    harga_beli_modal,
    biaya_pakan_per_hari,
    lama_pemeliharaan_hari,
    biaya_obat_vitamin,
    biaya_transportasi,
    biaya_lain_lain,
):
    """Menghitung biaya pemeliharaan, total modal, estimasi keuntungan, dan ROI."""
    harga_beli_modal = max(0, float(harga_beli_modal or 0))
    biaya_pakan_per_hari = max(0, float(biaya_pakan_per_hari or 0))
    lama_pemeliharaan_hari = max(0, float(lama_pemeliharaan_hari or 0))
    biaya_obat_vitamin = max(0, float(biaya_obat_vitamin or 0))
    biaya_transportasi = max(0, float(biaya_transportasi or 0))
    biaya_lain_lain = max(0, float(biaya_lain_lain or 0))

    biaya_pakan_total = biaya_pakan_per_hari * lama_pemeliharaan_hari
    total_biaya_pemeliharaan = (
        biaya_pakan_total
        + biaya_obat_vitamin
        + biaya_transportasi
        + biaya_lain_lain
    )
    total_modal = harga_beli_modal + total_biaya_pemeliharaan
    estimasi_keuntungan = float(nilai_jual or 0) - total_modal
    roi_percent = (estimasi_keuntungan / total_modal * 100) if total_modal > 0 else 0

    return {
        "biaya_pakan_total": biaya_pakan_total,
        "total_biaya_pemeliharaan": total_biaya_pemeliharaan,
        "total_modal": total_modal,
        "estimasi_keuntungan": estimasi_keuntungan,
        "roi_percent": roi_percent,
    }


def calculate_butcher_metrics(
    karkas_data,
    harga_beli_ternak,
    harga_jual_daging,
    harga_jual_tulang_lemak,
    harga_jual_non_karkas,
    biaya_pemotongan,
    biaya_transportasi,
    biaya_tenaga_kerja,
    biaya_es_penyimpanan,
    biaya_sewa_retribusi,
    biaya_lain_lain,
    target_margin_percent,
):
    """Menghitung estimasi omzet, biaya, profit, ROI, dan harga beli maksimal untuk jagal."""
    meat_weight = float(karkas_data.get("meat_weight", 0) or 0)
    bone_fat_weight = float(karkas_data.get("bone_and_fat_weight", 0) or 0)
    non_karkas_weights = karkas_data.get("non_karkas_weights", {}) or {}
    non_karkas_total = sum(float(value or 0) for value in non_karkas_weights.values())

    harga_beli_ternak = max(0, float(harga_beli_ternak or 0))
    harga_jual_daging = max(0, float(harga_jual_daging or 0))
    harga_jual_tulang_lemak = max(0, float(harga_jual_tulang_lemak or 0))
    harga_jual_non_karkas = max(0, float(harga_jual_non_karkas or 0))

    biaya_pemotongan = max(0, float(biaya_pemotongan or 0))
    biaya_transportasi = max(0, float(biaya_transportasi or 0))
    biaya_tenaga_kerja = max(0, float(biaya_tenaga_kerja or 0))
    biaya_es_penyimpanan = max(0, float(biaya_es_penyimpanan or 0))
    biaya_sewa_retribusi = max(0, float(biaya_sewa_retribusi or 0))
    biaya_lain_lain = max(0, float(biaya_lain_lain or 0))
    target_margin_percent = max(0, float(target_margin_percent or 0))

    omzet_daging = meat_weight * harga_jual_daging
    omzet_tulang_lemak = bone_fat_weight * harga_jual_tulang_lemak
    omzet_non_karkas = non_karkas_total * harga_jual_non_karkas
    omzet_total = omzet_daging + omzet_tulang_lemak + omzet_non_karkas

    biaya_operasional = (
        biaya_pemotongan
        + biaya_transportasi
        + biaya_tenaga_kerja
        + biaya_es_penyimpanan
        + biaya_sewa_retribusi
        + biaya_lain_lain
    )

    total_modal = harga_beli_ternak + biaya_operasional
    profit = omzet_total - total_modal
    roi_percent = (profit / total_modal * 100) if total_modal > 0 else 0

    target_profit = omzet_total * (target_margin_percent / 100)
    break_even_buy_price = omzet_total - biaya_operasional
    max_buy_price = break_even_buy_price - target_profit

    if profit < 0:
        decision = "Berisiko Rugi"
        decision_note = "Estimasi omzet belum menutup harga beli dan biaya operasional."
    elif roi_percent < target_margin_percent:
        decision = "Perlu Negosiasi"
        decision_note = "Masih profit, tetapi belum mencapai target margin."
    else:
        decision = "Layak Dibeli"
        decision_note = "Estimasi profit sudah memenuhi target margin."

    return {
        "meat_weight": meat_weight,
        "bone_fat_weight": bone_fat_weight,
        "non_karkas_total": non_karkas_total,
        "omzet_daging": omzet_daging,
        "omzet_tulang_lemak": omzet_tulang_lemak,
        "omzet_non_karkas": omzet_non_karkas,
        "omzet_total": omzet_total,
        "biaya_operasional": biaya_operasional,
        "total_modal": total_modal,
        "profit": profit,
        "roi_percent": roi_percent,
        "target_profit": target_profit,
        "break_even_buy_price": break_even_buy_price,
        "max_buy_price": max_buy_price,
        "decision": decision,
        "decision_note": decision_note,
    }


def create_butcher_component_dataframe(karkas_data, harga_jual_daging, harga_jual_tulang_lemak, harga_jual_non_karkas):
    """Membuat tabel ringkas komponen hasil potong untuk jagal."""
    meat_weight = float(karkas_data.get("meat_weight", 0) or 0)
    bone_fat_weight = float(karkas_data.get("bone_and_fat_weight", 0) or 0)
    non_karkas_weights = karkas_data.get("non_karkas_weights", {}) or {}
    non_karkas_total = sum(float(value or 0) for value in non_karkas_weights.values())

    rows = [
        {
            "Komponen": "Daging bersih",
            "Estimasi Berat (kg)": round(meat_weight, 2),
            "Harga/kg": format_rupiah(harga_jual_daging),
            "Estimasi Nilai": format_rupiah(meat_weight * harga_jual_daging),
        },
        {
            "Komponen": "Tulang & lemak karkas",
            "Estimasi Berat (kg)": round(bone_fat_weight, 2),
            "Harga/kg": format_rupiah(harga_jual_tulang_lemak),
            "Estimasi Nilai": format_rupiah(bone_fat_weight * harga_jual_tulang_lemak),
        },
        {
            "Komponen": "Non-karkas gabungan",
            "Estimasi Berat (kg)": round(non_karkas_total, 2),
            "Harga/kg": format_rupiah(harga_jual_non_karkas),
            "Estimasi Nilai": format_rupiah(non_karkas_total * harga_jual_non_karkas),
        },
    ]

    return pd.DataFrame(rows)


def create_non_karkas_detail_dataframe(karkas_data, harga_jual_non_karkas):
    """Membuat tabel detail komponen non-karkas untuk jagal."""
    rows = []
    for component, weight in (karkas_data.get("non_karkas_weights", {}) or {}).items():
        rows.append({
            "Komponen Non-Karkas": component,
            "Estimasi Berat (kg)": round(float(weight or 0), 2),
            "Estimasi Nilai": format_rupiah(float(weight or 0) * harga_jual_non_karkas),
        })
    return pd.DataFrame(rows)


def generate_butcher_recommendations(metrics):
    """Membuat rekomendasi keputusan untuk jagal."""
    recommendations = [
        f"Keputusan awal: {metrics['decision']}. {metrics['decision_note']}",
        f"Harga beli impas maksimal sekitar {format_rupiah(metrics['break_even_buy_price'])}.",
        f"Harga beli maksimal agar memenuhi target margin sekitar {format_rupiah(max(0, metrics['max_buy_price']))}.",
    ]

    if metrics["profit"] < 0:
        recommendations.append("Disarankan negosiasi harga beli, naikkan harga jual, atau cek ulang estimasi hasil potong.")
    elif metrics["roi_percent"] < 10:
        recommendations.append("Margin masih tipis. Tambahkan cadangan risiko untuk susut, kerusakan, atau harga pasar turun.")
    else:
        recommendations.append("Margin relatif aman berdasarkan harga dan biaya yang dimasukkan.")

    return recommendations





def get_efficiency_category(value, thresholds):
    """Mengubah nilai numerik menjadi kategori efisiensi sederhana."""
    low, medium, high = thresholds
    if value >= high:
        return "Sangat Baik"
    if value >= medium:
        return "Baik"
    if value >= low:
        return "Cukup"
    return "Rendah"


def calculate_operational_insights(
    berat_badan,
    karkas_data,
    accuracy_score,
    bcs_option,
    jagal_metrics=None,
    jenis_ternak=None,
    bangsa_ternak=None,
):
    """Membuat insight otomatis dari berat, karkas, BCS, akurasi input, dan metrik jagal."""
    breed_profile = get_breed_business_profile(jenis_ternak, bangsa_ternak) if jenis_ternak and bangsa_ternak else None
    berat_badan = max(0.0, float(berat_badan or 0))
    karkas_weight = float(karkas_data.get("karkas_weight", 0) or 0)
    meat_weight = float(karkas_data.get("meat_weight", 0) or 0)
    bone_fat_weight = float(karkas_data.get("bone_and_fat_weight", 0) or 0)
    non_karkas_total = sum(float(value or 0) for value in (karkas_data.get("non_karkas_weights", {}) or {}).values())

    karkas_yield = (karkas_weight / berat_badan * 100) if berat_badan > 0 else 0
    meat_yield_live = (meat_weight / berat_badan * 100) if berat_badan > 0 else 0
    meat_yield_carcass = (meat_weight / karkas_weight * 100) if karkas_weight > 0 else 0
    bone_fat_ratio = (bone_fat_weight / karkas_weight * 100) if karkas_weight > 0 else 0
    non_karkas_ratio = (non_karkas_total / berat_badan * 100) if berat_badan > 0 else 0

    karkas_category = get_efficiency_category(karkas_yield, (45, 50, 55))
    meat_category = get_efficiency_category(meat_yield_live, (30, 35, 40))
    accuracy_category = get_efficiency_category(float(accuracy_score or 0), (55, 70, 85))

    risk_notes = []
    opportunity_notes = []

    if accuracy_score < 70:
        risk_notes.append("Skor akurasi input belum tinggi; hasil estimasi sebaiknya diverifikasi ulang dengan pengukuran ulang atau timbangan.")
    else:
        opportunity_notes.append("Kualitas input cukup baik untuk dipakai sebagai estimasi awal.")

    if bcs_option in ["1 - Sangat Kurus", "2 - Kurus"]:
        risk_notes.append("BCS kurus berpotensi menurunkan hasil daging aktual dibandingkan angka estimasi.")
    elif bcs_option in ["4 - Gemuk", "5 - Sangat Gemuk"]:
        risk_notes.append("BCS gemuk dapat meningkatkan proporsi lemak, sehingga daging bersih perlu dibaca hati-hati.")
    elif bcs_option == "3 - Sedang/Ideal":
        opportunity_notes.append("BCS sedang/ideal mendukung hasil estimasi yang lebih stabil.")

    if karkas_yield < 48:
        risk_notes.append("Persentase karkas relatif rendah; cek kondisi tubuh, umur, dan bangsa ternak.")
    elif karkas_yield >= 55:
        opportunity_notes.append("Persentase karkas relatif baik; potensi hasil potong cukup menarik.")

    if meat_yield_live < 34:
        risk_notes.append("Estimasi daging bersih terhadap bobot hidup relatif rendah; margin jagal perlu lebih hati-hati.")
    elif meat_yield_live >= 40:
        opportunity_notes.append("Estimasi daging bersih terhadap bobot hidup relatif tinggi.")

    if jagal_metrics:
        if jagal_metrics.get("profit", 0) < 0:
            risk_notes.append("Simulasi jagal menunjukkan potensi rugi pada harga dan biaya yang dimasukkan.")
        elif jagal_metrics.get("roi_percent", 0) < 10:
            risk_notes.append("ROI jagal masih tipis; sisakan ruang negosiasi harga beli.")
        else:
            opportunity_notes.append("ROI jagal relatif baik berdasarkan simulasi harga dan biaya saat ini.")

    if breed_profile:
        risk_notes.append("Risiko khusus bangsa: " + "; ".join(breed_profile.get("risks", [])))
        opportunity_notes.append(f"Profil bangsa: {breed_profile.get('market_position', '')}")
        opportunity_notes.append(f"Sudut pandang jagal untuk {bangsa_ternak}: {breed_profile.get('butcher_view', '')}")

    if not risk_notes:
        risk_notes.append("Tidak ada risiko besar yang terdeteksi dari parameter utama, tetapi hasil tetap bersifat estimasi.")

    if not opportunity_notes:
        opportunity_notes.append("Peluang dapat ditingkatkan dengan memperbaiki harga jual, menekan biaya, atau memilih ternak dengan BCS lebih ideal.")

    return {
        "karkas_yield": karkas_yield,
        "meat_yield_live": meat_yield_live,
        "meat_yield_carcass": meat_yield_carcass,
        "bone_fat_ratio": bone_fat_ratio,
        "non_karkas_ratio": non_karkas_ratio,
        "karkas_category": karkas_category,
        "meat_category": meat_category,
        "accuracy_category": accuracy_category,
        "breed_profile": breed_profile,
        "risk_notes": risk_notes,
        "opportunity_notes": opportunity_notes,
    }


def create_price_sensitivity_dataframe(
    karkas_data,
    harga_jual_daging,
    harga_jual_tulang_lemak,
    harga_jual_non_karkas,
    harga_beli_ternak,
    biaya_operasional,
):
    """Membuat tabel sensitivitas profit jika harga jual berubah."""
    rows = []
    for change_percent in [-10, -5, 0, 5, 10]:
        factor = 1 + (change_percent / 100)
        metrics = calculate_butcher_metrics(
            karkas_data=karkas_data,
            harga_beli_ternak=harga_beli_ternak,
            harga_jual_daging=harga_jual_daging * factor,
            harga_jual_tulang_lemak=harga_jual_tulang_lemak * factor,
            harga_jual_non_karkas=harga_jual_non_karkas * factor,
            biaya_pemotongan=biaya_operasional,
            biaya_transportasi=0,
            biaya_tenaga_kerja=0,
            biaya_es_penyimpanan=0,
            biaya_sewa_retribusi=0,
            biaya_lain_lain=0,
            target_margin_percent=10,
        )
        rows.append({
            "Perubahan Harga Jual": f"{change_percent:+.0f}%",
            "Estimasi Omzet": format_rupiah(metrics["omzet_total"]),
            "Estimasi Profit": format_rupiah(metrics["profit"]),
            "ROI": f"{metrics['roi_percent']:.1f}%",
            "Keputusan": metrics["decision"],
        })
    return pd.DataFrame(rows)


def create_shrinkage_sensitivity_dataframe(
    jagal_metrics,
    harga_jual_daging,
):
    """Membuat simulasi dampak susut daging terhadap profit."""
    meat_weight = float(jagal_metrics.get("meat_weight", 0) or 0)
    base_profit = float(jagal_metrics.get("profit", 0) or 0)
    rows = []

    for shrink_percent in [0, 2, 5, 8, 10]:
        shrink_weight = meat_weight * (shrink_percent / 100)
        profit_after_shrink = base_profit - (shrink_weight * harga_jual_daging)
        rows.append({
            "Susut Daging": f"{shrink_percent}%",
            "Estimasi Susut (kg)": round(shrink_weight, 2),
            "Penurunan Nilai": format_rupiah(shrink_weight * harga_jual_daging),
            "Profit Setelah Susut": format_rupiah(profit_after_shrink),
        })

    return pd.DataFrame(rows)


def create_decision_checklist(insights, jagal_metrics=None):
    """Membuat checklist tindakan lanjutan berdasarkan insight."""
    checklist = []

    if insights["accuracy_category"] in ["Rendah", "Cukup"]:
        checklist.append("Ukur ulang lingkar dada dan panjang badan minimal 2–3 kali.")
    else:
        checklist.append("Data pengukuran cukup layak; tetap simpan hasil sebagai estimasi, bukan angka final.")

    if insights["karkas_category"] in ["Rendah", "Cukup"]:
        checklist.append("Cek kembali BCS, umur, dan kondisi fisik karena potensi karkas belum optimal.")
    else:
        checklist.append("Persentase karkas cukup menarik untuk dipertimbangkan dalam transaksi.")

    if jagal_metrics:
        if jagal_metrics.get("decision") == "Berisiko Rugi":
            checklist.append("Prioritas: turunkan harga beli atau batalkan transaksi jika harga jual tidak bisa naik.")
        elif jagal_metrics.get("decision") == "Perlu Negosiasi":
            checklist.append("Negosiasikan harga beli mendekati harga beli maksimal target margin.")
        else:
            checklist.append("Transaksi relatif layak, tetapi tetap sisakan cadangan risiko susut dan biaya tambahan.")

    checklist.append("Cocokkan kembali harga daging, tulang/lemak, dan non-karkas dengan pasar setempat.")
    return checklist



def calculate_trader_resale_score(
    berat_badan,
    jenis_ternak,
    status_ukuran,
    bcs_option,
    accuracy_score,
    trader_margin,
    trader_roi,
    bangsa_ternak=None,
):
    """Menghitung skor daya jual ternak dari kacamata blantik."""
    breed_profile = get_breed_business_profile(jenis_ternak, bangsa_ternak) if bangsa_ternak else None
    score = 50

    if accuracy_score >= 85:
        score += 15
    elif accuracy_score >= 70:
        score += 10
    elif accuracy_score >= 55:
        score += 4
    else:
        score -= 8

    if status_ukuran in ["Normal", "Besar"]:
        score += 12
    elif status_ukuran == "Sangat Besar":
        score += 6
    else:
        score -= 5

    if bcs_option == "3 - Sedang/Ideal":
        score += 15
    elif bcs_option in ["2 - Kurus", "4 - Gemuk"]:
        score += 5
    elif bcs_option in ["1 - Sangat Kurus", "5 - Sangat Gemuk"]:
        score -= 8

    if trader_margin > 0:
        score += 8
    else:
        score -= 12

    if trader_roi >= 15:
        score += 10
    elif trader_roi >= 8:
        score += 6
    elif trader_roi > 0:
        score += 2
    else:
        score -= 8

    if jenis_ternak == "Sapi" and berat_badan >= 250:
        score += 5
    elif jenis_ternak in ["Kambing", "Domba"] and berat_badan >= 25:
        score += 5

    if breed_profile:
        score += int(breed_profile.get("liquidity_bonus", 5)) - 5
        if breed_profile.get("premium_factor", 1.0) >= 1.10 and trader_margin > 0:
            score += 4
        if breed_profile.get("liquidity_bonus", 5) <= 5 and trader_margin <= 0:
            score -= 4

    score = max(0, min(100, int(round(score))))

    if score >= 85:
        category = "Sangat Mudah Dijual"
    elif score >= 70:
        category = "Mudah Dijual"
    elif score >= 55:
        category = "Cukup"
    else:
        category = "Sulit Dijual"

    return score, category


def determine_buyer_segments(jenis_ternak, berat_badan, bcs_option, status_ukuran, bangsa_ternak=None):
    """Menentukan segmentasi calon pembeli potensial."""
    profile = get_breed_business_profile(jenis_ternak, bangsa_ternak) if bangsa_ternak else None
    segments = list(profile.get("primary_buyers", [])) if profile else []

    if jenis_ternak == "Sapi":
        if berat_badan >= 250 and bcs_option in ["2 - Kurus", "3 - Sedang/Ideal", "4 - Gemuk", "Tidak dinilai"]:
            segments.append("Pembeli kurban")
        if berat_badan >= 300:
            segments.append("Jagal / pemotong")
        if bcs_option in ["1 - Sangat Kurus", "2 - Kurus"] and status_ukuran in ["Normal", "Besar"]:
            segments.append("Peternak penggemukan")
        segments.append("Pedagang pasar hewan")
    else:
        if berat_badan >= 22 and bcs_option in ["2 - Kurus", "3 - Sedang/Ideal", "4 - Gemuk", "Tidak dinilai"]:
            segments.append("Pembeli kurban")
        if berat_badan >= 25:
            segments.append("Jagal kecil / pedagang daging")
        if bcs_option in ["1 - Sangat Kurus", "2 - Kurus"]:
            segments.append("Peternak penggemukan")
        segments.append("Pedagang pasar hewan")

    # Hapus duplikat sambil menjaga urutan
    unique_segments = []
    for segment in segments:
        if segment not in unique_segments:
            unique_segments.append(segment)

    return unique_segments


def determine_trader_strategy(bcs_option, trader_margin, trader_roi, resale_score, status_ukuran):
    """Menentukan strategi jual dari kacamata blantik."""
    if trader_margin < 0:
        return "Tahan / Jangan Deal", "Margin masih negatif pada harga dan biaya yang dimasukkan."
    if bcs_option in ["1 - Sangat Kurus", "2 - Kurus"] and status_ukuran in ["Normal", "Besar"]:
        return "Penggemukan Lanjutan", "BCS masih bisa dinaikkan sehingga ada peluang peningkatan nilai jual."
    if resale_score >= 75 and trader_roi >= 10:
        return "Jual Cepat", "Daya jual dan margin cukup baik sehingga cocok untuk perputaran cepat."
    if trader_roi < 8:
        return "Perlu Negosiasi", "ROI masih tipis; harga beli atau biaya operasional perlu ditekan."
    return "Tahan 2–4 Minggu", "Masih ada peluang optimasi harga jual atau kondisi tubuh sebelum dijual."


def calculate_trader_insights(
    berat_badan,
    jenis_ternak,
    bangsa_ternak,
    status_ukuran,
    bcs_option,
    accuracy_score,
    harga_beli,
    harga_jual_per_kg,
    biaya_angkut,
    biaya_pakan_harian,
    lama_tahan_hari,
    biaya_kandang,
    biaya_retribusi,
    biaya_tenaga_bantu,
    biaya_lain,
    target_margin_percent,
):
    """Menghitung insight transaksi dari kacamata blantik ternak."""
    breed_profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)
    berat_badan = max(0, float(berat_badan or 0))
    harga_beli = max(0, float(harga_beli or 0))
    harga_jual_per_kg = max(0, float(harga_jual_per_kg or 0))
    biaya_angkut = max(0, float(biaya_angkut or 0))
    biaya_pakan_harian = max(0, float(biaya_pakan_harian or 0))
    lama_tahan_hari = max(0, float(lama_tahan_hari or 0))
    biaya_kandang = max(0, float(biaya_kandang or 0))
    biaya_retribusi = max(0, float(biaya_retribusi or 0))
    biaya_tenaga_bantu = max(0, float(biaya_tenaga_bantu or 0))
    biaya_lain = max(0, float(biaya_lain or 0))
    target_margin_percent = max(0, float(target_margin_percent or 0))

    estimasi_harga_jual = berat_badan * harga_jual_per_kg
    biaya_pakan_total = biaya_pakan_harian * lama_tahan_hari
    total_biaya_tambahan = (
        biaya_angkut
        + biaya_pakan_total
        + biaya_kandang
        + biaya_retribusi
        + biaya_tenaga_bantu
        + biaya_lain
    )
    total_modal = harga_beli + total_biaya_tambahan
    margin_bersih = estimasi_harga_jual - total_modal
    roi = (margin_bersih / total_modal * 100) if total_modal > 0 else 0

    target_profit = estimasi_harga_jual * (target_margin_percent / 100)
    harga_beli_impas = estimasi_harga_jual - total_biaya_tambahan
    harga_beli_maksimal = harga_beli_impas - target_profit
    harga_beli_ideal = harga_beli_maksimal * 0.95 if harga_beli_maksimal > 0 else 0

    resale_score, resale_category = calculate_trader_resale_score(
        berat_badan,
        jenis_ternak,
        status_ukuran,
        bcs_option,
        accuracy_score,
        margin_bersih,
        roi,
        bangsa_ternak=bangsa_ternak,
    )

    buyer_segments = determine_buyer_segments(
        jenis_ternak,
        berat_badan,
        bcs_option,
        status_ukuran,
        bangsa_ternak=bangsa_ternak,
    )

    strategy, strategy_note = determine_trader_strategy(
        bcs_option,
        margin_bersih,
        roi,
        resale_score,
        status_ukuran,
    )

    if breed_profile.get("fattening_fit", 5) >= 8 and strategy == "Tahan 2–4 Minggu":
        strategy = "Penggemukan Lanjutan"
        strategy_note = f"Profil {bangsa_ternak} mendukung penggemukan jika biaya pakan terkendali."
    elif breed_profile.get("liquidity_bonus", 5) >= 8 and margin_bersih > 0:
        strategy = "Jual Cepat"
        strategy_note = f"Profil {bangsa_ternak} relatif likuid, cocok untuk perputaran cepat."


    if margin_bersih < 0:
        decision = "Berisiko Rugi"
        decision_note = "Harga beli dan biaya tambahan lebih besar dari estimasi harga jual."
    elif roi < target_margin_percent:
        decision = "Perlu Negosiasi"
        decision_note = "Masih ada margin, tetapi belum mencapai target margin."
    elif resale_score < 55:
        decision = "Tahan Dulu"
        decision_note = "Margin bisa positif, tetapi daya jual masih perlu diperbaiki."
    else:
        decision = "Layak Dibeli"
        decision_note = "Margin, daya jual, dan strategi masih cukup mendukung transaksi."

    risk_level = "Rendah"
    risk_notes = []
    if margin_bersih < 0:
        risk_level = "Tinggi"
        risk_notes.append("Margin bersih negatif.")
    if roi < target_margin_percent:
        risk_level = "Sedang" if risk_level != "Tinggi" else risk_level
        risk_notes.append("ROI belum mencapai target margin.")
    if accuracy_score < 70:
        risk_level = "Sedang" if risk_level != "Tinggi" else risk_level
        risk_notes.append("Skor akurasi input belum kuat.")
    if bcs_option in ["1 - Sangat Kurus", "5 - Sangat Gemuk"]:
        risk_level = "Sedang" if risk_level != "Tinggi" else risk_level
        risk_notes.append("BCS ekstrem dapat menyulitkan jual ulang.")
    if breed_profile.get("risks"):
        risk_notes.append("Risiko spesifik bangsa: " + "; ".join(breed_profile.get("risks", [])))

    if not risk_notes:
        risk_notes.append("Risiko utama relatif terkendali berdasarkan input saat ini.")

    return {
        "estimasi_harga_jual": estimasi_harga_jual,
        "biaya_pakan_total": biaya_pakan_total,
        "total_biaya_tambahan": total_biaya_tambahan,
        "total_modal": total_modal,
        "margin_bersih": margin_bersih,
        "roi": roi,
        "target_profit": target_profit,
        "harga_beli_impas": harga_beli_impas,
        "harga_beli_maksimal": harga_beli_maksimal,
        "harga_beli_ideal": harga_beli_ideal,
        "resale_score": resale_score,
        "resale_category": resale_category,
        "buyer_segments": buyer_segments,
        "breed_profile": breed_profile,
        "strategy": strategy,
        "strategy_note": strategy_note,
        "decision": decision,
        "decision_note": decision_note,
        "risk_level": risk_level,
        "risk_notes": risk_notes,
    }


def create_trader_sensitivity_dataframe(
    berat_badan,
    harga_beli,
    harga_jual_per_kg,
    total_biaya_tambahan,
):
    """Membuat simulasi sensitivitas margin blantik jika harga jual/kg berubah."""
    rows = []
    for change_percent in [-10, -5, 0, 5, 10]:
        adjusted_price = harga_jual_per_kg * (1 + change_percent / 100)
        estimated_sale = berat_badan * adjusted_price
        margin = estimated_sale - harga_beli - total_biaya_tambahan
        total_modal = harga_beli + total_biaya_tambahan
        roi = (margin / total_modal * 100) if total_modal > 0 else 0
        rows.append({
            "Perubahan Harga Jual/kg": f"{change_percent:+.0f}%",
            "Harga Jual/kg": format_rupiah(adjusted_price),
            "Estimasi Harga Jual": format_rupiah(estimated_sale),
            "Margin Bersih": format_rupiah(margin),
            "ROI": f"{roi:.1f}%",
        })

    return pd.DataFrame(rows)


def create_trader_checklist(trader_insights):
    """Membuat checklist tindakan untuk blantik ternak."""
    checklist = [
        "Cek fisik langsung: mata, hidung, kaki, nafsu makan, dan kondisi kulit.",
        "Verifikasi umur, riwayat kesehatan, dan status kepemilikan sebelum transaksi.",
        "Bandingkan harga/kg dengan harga pasar lokal pada hari transaksi.",
    ]

    if trader_insights["decision"] == "Berisiko Rugi":
        checklist.insert(0, "Jangan deal sebelum harga beli turun atau harga jual ulang lebih jelas.")
    elif trader_insights["decision"] == "Perlu Negosiasi":
        checklist.insert(0, "Negosiasikan harga beli mendekati batas harga beli maksimal.")
    elif trader_insights["decision"] == "Tahan Dulu":
        checklist.insert(0, "Pertimbangkan penggemukan/penahanan singkat sebelum jual ulang.")
    else:
        checklist.insert(0, "Deal masih layak dipertimbangkan jika pemeriksaan fisik sesuai.")

    if trader_insights["strategy"] == "Penggemukan Lanjutan":
        checklist.append("Siapkan pakan dan estimasi lama tahan agar biaya tidak menggerus margin.")
    elif trader_insights["strategy"] == "Jual Cepat":
        checklist.append("Prioritaskan calon pembeli yang sudah siap agar perputaran modal cepat.")

    return checklist

def calculate_input_accuracy_score(
    lingkar_dada,
    panjang_badan,
    berat_badan,
    jenis_ternak,
    bangsa_ternak,
    bcs_option="Tidak dinilai",
):
    """Menghitung skor kualitas input pengukuran dalam rentang 0-100."""
    score = 100
    notes = []

    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]

    ld_in_range = chest_range["min"] <= lingkar_dada <= chest_range["max"]
    pb_in_range = length_range["min"] <= panjang_badan <= length_range["max"]

    if not ld_in_range:
        score -= 18
        notes.append("Lingkar dada berada di luar rentang normal bangsa ternak.")
    else:
        notes.append("Lingkar dada berada dalam rentang normal.")

    if not pb_in_range:
        score -= 14
        notes.append("Panjang badan berada di luar rentang normal bangsa ternak.")
    else:
        notes.append("Panjang badan berada dalam rentang normal.")

    ld_mid = (chest_range["min"] + chest_range["max"]) / 2
    pb_mid = (length_range["min"] + length_range["max"]) / 2
    ld_ratio = lingkar_dada / ld_mid if ld_mid else 1
    pb_ratio = panjang_badan / pb_mid if pb_mid else 1

    if abs(ld_ratio - pb_ratio) > 0.25:
        score -= 12
        notes.append("Perbandingan lingkar dada dan panjang badan terlihat kurang seimbang.")
    else:
        notes.append("Perbandingan lingkar dada dan panjang badan masih seimbang.")

    if not validate_weight_result(berat_badan, jenis_ternak):
        score -= 20
        notes.append("Prediksi berat berada di luar rentang wajar jenis ternak.")
    else:
        notes.append("Prediksi berat masih berada dalam rentang wajar.")

    if bcs_option in ["1 - Sangat Kurus", "5 - Sangat Gemuk"]:
        score -= 12
        notes.append("BCS ekstrem dapat menurunkan akurasi estimasi.")
    elif bcs_option in ["2 - Kurus", "4 - Gemuk"]:
        score -= 6
        notes.append("BCS agak menyimpang dari ideal; hasil perlu dibaca sebagai estimasi.")
    elif bcs_option == "3 - Sedang/Ideal":
        notes.append("BCS ideal mendukung stabilitas estimasi.")
    else:
        score -= 3
        notes.append("BCS belum dinilai, sehingga akurasi kondisi tubuh belum terverifikasi.")

    score = max(0, min(100, int(round(score))))

    if score >= 85:
        category = "Sangat Baik"
    elif score >= 70:
        category = "Baik"
    elif score >= 55:
        category = "Cukup"
    else:
        category = "Perlu Cek Ulang"

    return score, category, notes


def estimate_dimensions_for_target_weight(
    target_weight,
    current_lingkar_dada,
    current_panjang_badan,
    jenis_ternak,
    bangsa_ternak,
    jenis_kelamin,
):
    """Mengestimasi LD dan PB untuk target berat dengan mempertahankan proporsi ukuran saat ini."""
    target_weight = max(0.0, float(target_weight or 0))
    current_lingkar_dada = max(0.1, float(current_lingkar_dada or 0.1))
    current_panjang_badan = max(0.1, float(current_panjang_badan or 0.1))

    if target_weight <= 0:
        return {
            "lingkar_dada": 0,
            "panjang_badan": 0,
            "estimated_weight": 0,
            "status": "Target belum valid",
            "note": "Masukkan target berat lebih dari 0 kg.",
        }

    low_scale = 0.40
    high_scale = 2.50

    for _ in range(60):
        mid_scale = (low_scale + high_scale) / 2
        candidate_ld = current_lingkar_dada * mid_scale
        candidate_pb = current_panjang_badan * mid_scale
        candidate_weight, _, _ = hitung_berat_badan(
            candidate_ld,
            candidate_pb,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )

        if candidate_weight < target_weight:
            low_scale = mid_scale
        else:
            high_scale = mid_scale

    scale = (low_scale + high_scale) / 2
    estimated_ld = current_lingkar_dada * scale
    estimated_pb = current_panjang_badan * scale
    estimated_weight, _, _ = hitung_berat_badan(
        estimated_ld,
        estimated_pb,
        jenis_ternak,
        bangsa_ternak,
        jenis_kelamin,
    )

    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]

    ld_ok = chest_range["min"] <= estimated_ld <= chest_range["max"]
    pb_ok = length_range["min"] <= estimated_pb <= length_range["max"]

    if ld_ok and pb_ok:
        status = "Realistis"
        note = "Perkiraan ukuran masih berada dalam rentang normal bangsa ternak."
    elif estimated_ld <= chest_range["max"] * 1.2 and estimated_pb <= length_range["max"] * 1.2:
        status = "Masih Mungkin"
        note = "Perkiraan ukuran sedikit di luar rentang normal; perlu verifikasi kondisi lapangan."
    else:
        status = "Kurang Realistis"
        note = "Target berat membutuhkan ukuran yang cukup jauh dari rentang umum bangsa ternak."

    return {
        "lingkar_dada": estimated_ld,
        "panjang_badan": estimated_pb,
        "estimated_weight": estimated_weight,
        "status": status,
        "note": note,
    }




def estimate_weight_from_breed_percentile(jenis_ternak, bangsa_ternak, jenis_kelamin, percentile):
    """Menghitung estimasi berat dari posisi persentil rentang LD/PB bangsa ternak."""
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]

    percentile = max(0.0, min(1.0, float(percentile)))
    ld = chest_range["min"] + ((chest_range["max"] - chest_range["min"]) * percentile)
    pb = length_range["min"] + ((length_range["max"] - length_range["min"]) * percentile)
    weight, _, _ = hitung_berat_badan(ld, pb, jenis_ternak, bangsa_ternak, jenis_kelamin)

    return {
        "lingkar_dada": ld,
        "panjang_badan": pb,
        "berat": weight,
    }


def get_breed_target_examples(jenis_ternak, bangsa_ternak, jenis_kelamin):
    """Membuat contoh target berat yang disesuaikan dengan rentang normal jenis + bangsa ternak."""
    profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)

    base_examples = [
        {
            "label": "Target Ringan",
            "percentile": 0.25,
            "tujuan": "Ternak ukuran bawah-menengah; cocok untuk evaluasi awal atau penggemukan lanjutan.",
        },
        {
            "label": "Target Standar",
            "percentile": 0.50,
            "tujuan": "Titik tengah rentang normal bangsa ternak; cocok sebagai target aman.",
        },
        {
            "label": "Target Optimal",
            "percentile": 0.75,
            "tujuan": "Ukuran atas-menengah; cocok untuk pasar yang mengutamakan bobot.",
        },
        {
            "label": "Target Maksimal Normal",
            "percentile": 0.95,
            "tujuan": "Mendekati batas atas normal; cocok jika pasar dan pakan mendukung.",
        },
    ]

    rows = []
    for item in base_examples:
        estimate = estimate_weight_from_breed_percentile(
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
            item["percentile"],
        )

        if profile.get("premium_factor", 1.0) >= 1.10 and item["label"] in ["Target Optimal", "Target Maksimal Normal"]:
            segment = "Premium / jagal / pembeli berbobot besar"
        elif profile.get("liquidity_bonus", 5) >= 8 and item["label"] in ["Target Standar", "Target Optimal"]:
            segment = "Pasar umum / kurban / jual cepat"
        elif profile.get("fattening_fit", 5) >= 8 and item["label"] in ["Target Standar", "Target Optimal"]:
            segment = "Penggemukan / jual ulang"
        else:
            segment = ", ".join(profile.get("primary_buyers", [])[:2])

        rows.append({
            "Contoh Target": item["label"],
            "Target Berat (kg)": round(estimate["berat"], 2),
            "Estimasi LD (cm)": round(estimate["lingkar_dada"], 1),
            "Estimasi PB (cm)": round(estimate["panjang_badan"], 1),
            "Segmen Cocok": segment,
            "Catatan": item["tujuan"],
        })

    return pd.DataFrame(rows)


def get_recommended_target_weight_by_breed(current_weight, jenis_ternak, bangsa_ternak, jenis_kelamin):
    """Memilih contoh target terdekat di atas berat saat ini berdasarkan bangsa ternak."""
    examples_df = get_breed_target_examples(jenis_ternak, bangsa_ternak, jenis_kelamin)
    current_weight = float(current_weight or 0)

    above_current = examples_df[examples_df["Target Berat (kg)"] > current_weight]
    if not above_current.empty:
        selected = above_current.iloc[0]
    else:
        selected = examples_df.iloc[-1].copy()
        selected["Target Berat (kg)"] = max(current_weight * 1.05, selected["Target Berat (kg)"])

    return float(selected["Target Berat (kg)"])


def create_breed_target_option_labels(examples_df):
    """Membuat pilihan label yang mudah dibaca untuk selectbox target."""
    labels = []
    for _, row in examples_df.iterrows():
        labels.append(
            f"{row['Contoh Target']} - {row['Target Berat (kg)']:.2f} kg"
        )
    return labels



def create_target_simulation_table(
    base_weight,
    lingkar_dada,
    panjang_badan,
    jenis_ternak,
    bangsa_ternak,
    jenis_kelamin,
):
    """Membuat tabel simulasi target berbasis jenis + bangsa ternak."""
    examples_df = get_breed_target_examples(jenis_ternak, bangsa_ternak, jenis_kelamin)

    rows = []
    for _, target_row in examples_df.iterrows():
        target = float(target_row["Target Berat (kg)"])
        estimate = estimate_dimensions_for_target_weight(
            target,
            lingkar_dada,
            panjang_badan,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )
        rows.append({
            "Contoh Target": target_row["Contoh Target"],
            "Target Berat (kg)": round(target, 2),
            "Estimasi LD (cm)": round(estimate["lingkar_dada"], 1),
            "Estimasi PB (cm)": round(estimate["panjang_badan"], 1),
            "Estimasi Ulang BB (kg)": round(estimate["estimated_weight"], 2),
            "Selisih dari BB Saat Ini (kg)": round(target - base_weight, 2),
            "Status": estimate["status"],
            "Segmen Cocok": target_row["Segmen Cocok"],
            "Catatan": target_row["Catatan"],
        })

    current_row = {
        "Contoh Target": "Posisi Saat Ini",
        "Target Berat (kg)": round(base_weight, 2),
        "Estimasi LD (cm)": round(lingkar_dada, 1),
        "Estimasi PB (cm)": round(panjang_badan, 1),
        "Estimasi Ulang BB (kg)": round(base_weight, 2),
        "Selisih dari BB Saat Ini (kg)": 0.0,
        "Status": "Aktual Input",
        "Segmen Cocok": "Pembanding",
        "Catatan": "Posisi berdasarkan pengukuran yang dimasukkan pengguna.",
    }

    result_df = pd.DataFrame([current_row] + rows)
    return result_df




def build_ai_prompt_from_results(
    prompt_mode,
    jenis_ternak,
    bangsa_ternak,
    jenis_kelamin,
    lingkar_dada,
    panjang_badan,
    berat_badan,
    bb_min,
    bb_max,
    margin_error,
    formula_name,
    formula_text,
    status_ukuran,
    status_note,
    bcs_option,
    accuracy_score,
    accuracy_category,
    karkas_data,
    breed_profile,
    target_table=None,
    harga_bobot_hidup=0,
    harga_karkas=0,
    harga_daging=0,
    nilai_hidup=0,
    nilai_karkas=0,
    nilai_daging=0,
    business_metrics=None,
    jagal_metrics=None,
    trader_insights=None,
    insights=None,
):
    """Menyusun prompt siap salin untuk AI lain berdasarkan hasil perhitungan aplikasi."""
    business_metrics = business_metrics or {}
    jagal_metrics = jagal_metrics or {}
    trader_insights = trader_insights or {}
    insights = insights or {}

    target_text = "Belum tersedia."
    if target_table is not None:
        try:
            target_text = target_table.to_string(index=False)
        except Exception:
            target_text = str(target_table)

    non_karkas_total = sum((karkas_data.get("non_karkas_weights", {}) or {}).values())

    role_map = {
        "Peternak": "Anda adalah konsultan peternakan ruminansia yang fokus pada manajemen bobot, BCS, pakan, kesehatan umum, dan strategi pemeliharaan.",
        "Jagal": "Anda adalah analis usaha jagal/pemotongan ternak yang fokus pada estimasi karkas, daging, susut, omzet, biaya, dan margin.",
        "Blantik": "Anda adalah analis transaksi blantik ternak yang fokus pada harga beli, harga jual ulang, daya jual, risiko transaksi, dan strategi negosiasi.",
        "Analisis Lengkap": "Anda adalah konsultan peternakan, jagal, dan perdagangan ternak yang mampu membaca data teknis, ekonomi, pasar, dan risiko transaksi secara terpadu.",
    }

    instruction_map = {
        "Peternak": """
Tugas Anda:
1. Jelaskan kondisi ternak dari sisi bobot, ukuran tubuh, BCS, dan skor akurasi input.
2. Berikan interpretasi apakah target berat realistis berdasarkan jenis dan bangsa ternak.
3. Berikan saran pemeliharaan umum: pakan, pengukuran ulang, pencatatan, dan hal yang perlu diperiksa.
4. Berikan strategi apakah ternak lebih cocok dijual sekarang, ditahan, atau digemukkan.
5. Berikan catatan risiko, terutama jika BCS ekstrem atau skor akurasi rendah.
""",
        "Jagal": """
Tugas Anda:
1. Analisis kelayakan ternak dari sudut pandang jagal.
2. Jelaskan potensi karkas, daging bersih, tulang/lemak, dan non-karkas.
3. Hitung ulang secara konseptual apakah omzet, biaya, profit, dan ROI masih masuk akal.
4. Berikan batas harga beli aman dan risiko susut.
5. Berikan rekomendasi: layak dipotong, perlu negosiasi harga, atau berisiko rugi.
""",
        "Blantik": """
Tugas Anda:
1. Analisis ternak dari sudut pandang blantik/pedagang ternak.
2. Jelaskan estimasi harga jual kembali, margin bersih, ROI, dan batas harga nego.
3. Nilai daya jual ternak berdasarkan jenis, bangsa, bobot, BCS, dan segmentasi pembeli.
4. Berikan strategi jual: jual cepat, tahan 2–4 minggu, penggemukan lanjutan, atau jangan deal.
5. Buat checklist negosiasi dan hal yang harus dicek sebelum transaksi.
""",
        "Analisis Lengkap": """
Tugas Anda:
1. Buat analisis lengkap dari sudut pandang peternak, jagal, dan blantik.
2. Rangkum kondisi bobot, ukuran, BCS, skor akurasi, target berat, hasil potong, ekonomi, profit, dan risiko.
3. Berikan rekomendasi tindakan prioritas.
4. Berikan skenario keputusan: jual sekarang, tahan/penggemukan, potong, atau negosiasi.
5. Buat tabel ringkas kesimpulan dan checklist pemeriksaan lapangan.
""",
    }

    mode = prompt_mode if prompt_mode in role_map else "Analisis Lengkap"

    prompt = f"""
{role_map[mode]}

Saya memiliki data hasil perhitungan aplikasi prediksi berat badan ternak. Tolong analisis data berikut secara detail, praktis, dan mudah dipahami untuk konteks Indonesia.

DATA IDENTITAS TERNAK
- Jenis ternak: {jenis_ternak}
- Bangsa ternak: {bangsa_ternak}
- Jenis kelamin: {jenis_kelamin}
- Lingkar dada: {lingkar_dada:.1f} cm
- Panjang badan: {panjang_badan:.1f} cm
- Rumus yang digunakan: {formula_name}
- Formula: {formula_text}

HASIL PREDIKSI BERAT
- Prediksi berat badan: {berat_badan:.2f} kg
- Rentang estimasi dengan margin error ±{margin_error}%: {bb_min:.2f}–{bb_max:.2f} kg
- Status ukuran: {status_ukuran}
- Catatan status ukuran: {status_note}
- BCS/kondisi tubuh: {bcs_option}
- Skor akurasi input: {accuracy_score}/100 ({accuracy_category})

PROFIL JENIS DAN BANGSA TERNAK
- Posisi pasar: {breed_profile.get('market_position', '-')}
- Pembeli potensial: {', '.join(breed_profile.get('primary_buyers', []))}
- Sudut pandang jagal: {breed_profile.get('butcher_view', '-')}
- Sudut pandang blantik: {breed_profile.get('trader_view', '-')}
- Strategi umum: {breed_profile.get('strategy', '-')}
- Risiko khusus bangsa: {'; '.join(breed_profile.get('risks', []))}
- Likuiditas pasar: {breed_profile.get('liquidity_bonus', '-')}/10
- Kesesuaian jagal: {breed_profile.get('butcher_fit', '-')}/10
- Kesesuaian penggemukan: {breed_profile.get('fattening_fit', '-')}/10
- Faktor premium harga: {breed_profile.get('premium_factor', '-')}

HASIL POTONG / KARKAS
- Persentase karkas: {karkas_data.get('karkas_percent', 0):.1f}%
- Estimasi berat karkas: {karkas_data.get('karkas_weight', 0):.2f} kg
- Persentase daging dari karkas: {karkas_data.get('meat_percent_of_carcass', 0):.1f}%
- Persentase daging dari bobot hidup: {karkas_data.get('meat_percent_of_body', 0):.1f}%
- Estimasi daging bersih: {karkas_data.get('meat_weight', 0):.2f} kg
- Estimasi tulang dan lemak karkas: {karkas_data.get('bone_and_fat_weight', 0):.2f} kg
- Estimasi non-karkas total: {non_karkas_total:.2f} kg

TARGET BERAT BERDASARKAN JENIS DAN BANGSA
{target_text}

DATA EKONOMI TERNAK
- Harga/kg bobot hidup: {format_rupiah(harga_bobot_hidup)}
- Estimasi nilai bobot hidup: {format_rupiah(nilai_hidup)}
- Harga/kg karkas: {format_rupiah(harga_karkas)}
- Estimasi nilai karkas: {format_rupiah(nilai_karkas)}
- Harga/kg daging: {format_rupiah(harga_daging)}
- Estimasi nilai daging: {format_rupiah(nilai_daging)}

DATA BIAYA & PROFIT PETERNAK
- Total biaya pemeliharaan: {format_rupiah(business_metrics.get('total_biaya_pemeliharaan', 0))}
- Total modal: {format_rupiah(business_metrics.get('total_modal', 0))}
- Estimasi keuntungan: {format_rupiah(business_metrics.get('estimasi_keuntungan', 0))}
- ROI: {business_metrics.get('roi_percent', 0):.1f}%

DATA JAGAL
- Omzet total jagal: {format_rupiah(jagal_metrics.get('omzet_total', 0))}
- Total modal jagal: {format_rupiah(jagal_metrics.get('total_modal', 0))}
- Profit jagal: {format_rupiah(jagal_metrics.get('profit', 0))}
- ROI jagal: {jagal_metrics.get('roi_percent', 0):.1f}%
- Harga beli impas jagal: {format_rupiah(jagal_metrics.get('break_even_buy_price', 0))}
- Harga beli maksimal target margin jagal: {format_rupiah(max(0, jagal_metrics.get('max_buy_price', 0)))}
- Keputusan jagal: {jagal_metrics.get('decision', '-')}

DATA BLANTIK
- Estimasi harga jual kembali: {format_rupiah(trader_insights.get('estimasi_harga_jual', 0))}
- Margin bersih blantik: {format_rupiah(trader_insights.get('margin_bersih', 0))}
- ROI blantik: {trader_insights.get('roi', 0):.1f}%
- Skor daya jual: {trader_insights.get('resale_score', 0)}/100 ({trader_insights.get('resale_category', '-')})
- Harga ideal beli: {format_rupiah(max(0, trader_insights.get('harga_beli_ideal', 0)))}
- Harga maksimal beli: {format_rupiah(max(0, trader_insights.get('harga_beli_maksimal', 0)))}
- Harga impas: {format_rupiah(max(0, trader_insights.get('harga_beli_impas', 0)))}
- Strategi blantik: {trader_insights.get('strategy', '-')}
- Keputusan blantik: {trader_insights.get('decision', '-')}
- Risiko transaksi blantik: {trader_insights.get('risk_level', '-')}
- Segmen pembeli: {', '.join(trader_insights.get('buyer_segments', [])) if trader_insights else '-'}

INSIGHT OTOMATIS APLIKASI
- Efisiensi karkas: {insights.get('karkas_yield', 0):.1f}%
- Daging terhadap bobot hidup: {insights.get('meat_yield_live', 0):.1f}%
- Daging terhadap karkas: {insights.get('meat_yield_carcass', 0):.1f}%
- Non-karkas terhadap bobot hidup: {insights.get('non_karkas_ratio', 0):.1f}%
- Kategori efisiensi karkas: {insights.get('karkas_category', '-')}
- Kategori daging: {insights.get('meat_category', '-')}

{instruction_map[mode]}

Format jawaban yang saya inginkan:
1. Ringkasan kondisi ternak dalam 5–7 poin.
2. Tabel interpretasi angka utama.
3. Insight berdasarkan jenis dan bangsa ternak.
4. Risiko utama dan cara menguranginya.
5. Rekomendasi tindakan praktis.
6. Checklist pemeriksaan lapangan sebelum keputusan.
7. Kesimpulan akhir dengan status: Aman / Perlu Cek Ulang / Perlu Negosiasi / Berisiko.

Catatan penting:
- Jangan menganggap angka ini sebagai hasil timbangan pasti.
- Jelaskan jika ada data yang perlu diverifikasi ulang.
- Gunakan bahasa Indonesia yang jelas, praktis, dan mudah dipahami peternak.
- Jika memberikan rekomendasi harga, sebutkan bahwa harga pasar lokal tetap harus dicek.
""".strip()

    return prompt

def generate_recommendations(berat_badan, lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin, kelas_pasar=None, margin_error=None, estimasi_keuntungan=None, bcs_option=None, accuracy_score=None):
    """Membuat rekomendasi otomatis berdasarkan hasil prediksi."""
    status, status_note = get_size_status(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak)
    recommendations = [
        f"Status ukuran ternak: {status}. {status_note}",
        "Lakukan pengukuran 2–3 kali, lalu gunakan nilai rata-rata agar prediksi lebih stabil.",
        "Untuk transaksi bernilai besar, tetap gunakan timbangan ternak yang terkalibrasi sebagai pembanding.",
    ]

    if kelas_pasar:
        recommendations.append(f"Kelas pasar yang digunakan untuk estimasi harga: {kelas_pasar}.")

    if margin_error:
        recommendations.append(f"Gunakan rentang estimasi ±{margin_error}% agar hasil tidak dianggap sebagai angka pasti.")

    if estimasi_keuntungan is not None:
        if estimasi_keuntungan >= 0:
            recommendations.append("Estimasi usaha masih positif berdasarkan harga jual dan biaya yang dimasukkan.")
        else:
            recommendations.append("Estimasi usaha negatif. Cek kembali harga beli, biaya pakan, dan target harga jual.")

    if bcs_option:
        recommendations.append(f"BCS/kondisi tubuh: {bcs_option}. {BCS_NOTES.get(bcs_option, '')}")

    if accuracy_score is not None:
        if accuracy_score >= 85:
            recommendations.append("Skor akurasi input sangat baik. Data pengukuran relatif layak digunakan sebagai estimasi awal.")
        elif accuracy_score >= 70:
            recommendations.append("Skor akurasi input baik, tetapi tetap lakukan pengukuran ulang untuk memastikan.")
        else:
            recommendations.append("Skor akurasi input belum optimal. Disarankan cek ulang LD, PB, dan kondisi tubuh ternak.")

    if berat_badan <= 0:
        recommendations.insert(0, "Hasil prediksi belum wajar. Cek kembali rumus, satuan, dan data input.")
    elif jenis_ternak == "Sapi" and berat_badan < 150:
        recommendations.insert(0, "Prediksi berat sapi cukup rendah. Pastikan data lingkar dada dan panjang badan tidak tertukar.")
    elif jenis_ternak in ["Kambing", "Domba"] and berat_badan < 10:
        recommendations.insert(0, "Prediksi berat ruminansia kecil sangat rendah. Cek ulang satuan pengukuran dalam cm.")

    if jenis_kelamin == "Jantan":
        recommendations.append("Pada ternak jantan, variasi bobot dapat lebih besar karena faktor pertumbuhan dan kondisi tubuh.")
    else:
        recommendations.append("Pada ternak betina, kondisi reproduksi dan kebuntingan dapat mempengaruhi hasil estimasi ukuran tubuh.")

    return recommendations


def create_pdf_report(report_data):
    """Membuat laporan PDF sederhana untuk diunduh dari Streamlit."""
    buffer = BytesIO()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "Laporan Prediksi Berat Badan Ternak")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Tanggal: {report_data.get('tanggal', '-')}")
    y -= 0.7 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Data Ternak")
    y -= 0.5 * cm

    c.setFont("Helvetica", 10)
    rows = [
        ("1. Jenis Ternak", report_data.get("jenis_ternak", "-")),
        ("Bangsa Ternak", report_data.get("bangsa_ternak", "-")),
        ("3. Jenis Kelamin", report_data.get("jenis_kelamin", "-")),
        ("BCS / Kondisi Tubuh", report_data.get("bcs_option", "-")),
        ("Skor Akurasi Input", f"{report_data.get('accuracy_score', 0)}/100 ({report_data.get('accuracy_category', '-')})"),
        ("Kelas Pasar", report_data.get("kelas_pasar", "-")),
        ("Margin Error", f"±{report_data.get('margin_error', 0)}%"),
        ("Lingkar Dada", f"{report_data.get('lingkar_dada', 0):.1f} cm"),
        ("Panjang Badan", f"{report_data.get('panjang_badan', 0):.1f} cm"),
        ("Rumus", report_data.get("formula_name", "-")),
        ("Prediksi Berat Badan", f"{report_data.get('berat_badan', 0):.2f} kg"),
        ("Rentang Berat Badan", f"{report_data.get('bb_min', 0):.2f} - {report_data.get('bb_max', 0):.2f} kg"),
    ]

    for label, value in rows:
        c.drawString(2 * cm, y, f"{label}:")
        c.drawString(7 * cm, y, str(value))
        y -= 0.45 * cm

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Estimasi Karkas dan Daging")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)

    rows = [
        ("Berat Karkas", f"{report_data.get('karkas_weight', 0):.2f} kg"),
        ("Berat Daging", f"{report_data.get('meat_weight', 0):.2f} kg"),
        ("Berat Tulang & Lemak Karkas", f"{report_data.get('bone_and_fat_weight', 0):.2f} kg"),
        ("Status Ukuran", report_data.get("status_ukuran", "-")),
    ]

    for label, value in rows:
        c.drawString(2 * cm, y, f"{label}:")
        c.drawString(7 * cm, y, str(value))
        y -= 0.45 * cm

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Estimasi Nilai Ekonomi")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)

    rows = [
        ("Harga/kg Bobot Hidup", format_rupiah(report_data.get("harga_hidup", 0))),
        ("Estimasi Nilai Bobot Hidup", format_rupiah(report_data.get("nilai_hidup", 0))),
        ("Harga/kg Karkas", format_rupiah(report_data.get("harga_karkas", 0))),
        ("Estimasi Nilai Karkas", format_rupiah(report_data.get("nilai_karkas", 0))),
        ("Harga/kg Daging", format_rupiah(report_data.get("harga_daging", 0))),
        ("Estimasi Nilai Daging", format_rupiah(report_data.get("nilai_daging", 0))),
        ("Total Biaya Pemeliharaan", format_rupiah(report_data.get("total_biaya_pemeliharaan", 0))),
        ("Total Modal", format_rupiah(report_data.get("total_modal", 0))),
        ("Estimasi Keuntungan", format_rupiah(report_data.get("estimasi_keuntungan", 0))),
        ("ROI", f"{report_data.get('roi_percent', 0):.1f}%"),
    ]

    for label, value in rows:
        c.drawString(2 * cm, y, f"{label}:")
        c.drawString(7 * cm, y, str(value))
        y -= 0.45 * cm

    y -= 0.4 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Catatan")
    y -= 0.5 * cm
    c.setFont("Helvetica", 9)
    notes = [
        "Hasil aplikasi adalah estimasi berbasis rumus dan data rata-rata.",
        "Hasil aktual dapat berbeda karena umur, kondisi tubuh, pakan, kesehatan, dan metode pengukuran.",
        "Gunakan timbangan ternak terkalibrasi untuk keputusan transaksi besar atau penelitian.",
    ]
    for note in notes:
        c.drawString(2 * cm, y, f"- {note}")
        y -= 0.4 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def process_batch_dataframe(df):
    """Menghitung prediksi banyak ternak dari dataframe upload CSV/XLSX."""
    required_cols = [
        "Jenis Ternak",
        "Bangsa Ternak",
        "Jenis Kelamin",
        "Lingkar Dada",
        "Panjang Badan",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Kolom wajib belum lengkap: " + ", ".join(missing_cols))

    results = []
    for idx, row in df.iterrows():
        try:
            jenis = str(row["Jenis Ternak"]).strip()
            bangsa = str(row["Bangsa Ternak"]).strip()
            kelamin = str(row["Jenis Kelamin"]).strip()
            ld = float(row["Lingkar Dada"])
            pb = float(row["Panjang Badan"])

            berat, formula_name, formula_text = hitung_berat_badan(ld, pb, jenis, bangsa, kelamin)
            karkas = hitung_komponen_karkas(berat, jenis, bangsa, kelamin)
            status, _ = get_size_status(ld, pb, jenis, bangsa)

            bcs_batch = str(row.get("BCS / Kondisi Tubuh", "Tidak dinilai")).strip()
            if not bcs_batch or bcs_batch.lower() == "nan":
                bcs_batch = "Tidak dinilai"
            if bcs_batch not in BCS_OPTIONS:
                bcs_batch = "Tidak dinilai"

            accuracy_score_batch, accuracy_category_batch, _ = calculate_input_accuracy_score(
                ld,
                pb,
                berat,
                jenis,
                bangsa,
                bcs_batch,
            )

            kelas_input = str(row.get("Kelas Pasar", "Otomatis")).strip()
            if not kelas_input or kelas_input.lower() == "nan":
                kelas_input = "Otomatis"
            kelas_pasar_batch, _, kelas_multiplier_batch = get_market_class(status, kelas_input)

            price_defaults = get_latest_price_defaults(jenis, bangsa)
            price_defaults = apply_market_class_to_prices(
                price_defaults,
                kelas_multiplier_batch,
                kelas_pasar_batch,
            )

            harga_hidup = clean_price_value(row.get("Harga per Kg", 0), price_defaults["harga_bobot_hidup"])
            harga_karkas_batch = clean_price_value(row.get("Harga per Kg Karkas", 0), price_defaults["harga_karkas"])
            harga_daging_batch = clean_price_value(row.get("Harga per Kg Daging", 0), price_defaults["harga_daging"])

            margin_error_batch = clean_price_value(row.get("Margin Error (%)", 10), 10)
            bb_min, bb_max = calculate_error_range(berat, margin_error_batch)

            nilai_ternak = berat * harga_hidup
            nilai_karkas = karkas["karkas_weight"] * harga_karkas_batch
            nilai_daging = karkas["meat_weight"] * harga_daging_batch

            harga_beli_modal_batch = clean_price_value(row.get("Harga Beli / Modal", 0), 0)
            biaya_pakan_per_hari_batch = clean_price_value(row.get("Biaya Pakan per Hari", 0), 0)
            lama_pemeliharaan_batch = clean_price_value(row.get("Lama Pemeliharaan (Hari)", 0), 0)
            biaya_obat_batch = clean_price_value(row.get("Biaya Obat/Vitamin", 0), 0)
            biaya_transport_batch = clean_price_value(row.get("Biaya Transportasi", 0), 0)
            biaya_lain_batch = clean_price_value(row.get("Biaya Lain-lain", 0), 0)

            business = calculate_maintenance_metrics(
                nilai_jual=nilai_ternak,
                harga_beli_modal=harga_beli_modal_batch,
                biaya_pakan_per_hari=biaya_pakan_per_hari_batch,
                lama_pemeliharaan_hari=lama_pemeliharaan_batch,
                biaya_obat_vitamin=biaya_obat_batch,
                biaya_transportasi=biaya_transport_batch,
                biaya_lain_lain=biaya_lain_batch,
            )

            results.append({
                "No": idx + 1,
                "Jenis Ternak": jenis,
                "Bangsa Ternak": bangsa,
                "Jenis Kelamin": kelamin,
                "Lingkar Dada (cm)": ld,
                "Panjang Badan (cm)": pb,
                "Rumus": formula_name,
                "Prediksi Berat (kg)": round(berat, 2),
                "BB Min (kg)": round(bb_min, 2),
                "BB Max (kg)": round(bb_max, 2),
                "Berat Karkas (kg)": round(karkas["karkas_weight"], 2),
                "Berat Daging (kg)": round(karkas["meat_weight"], 2),
                "Status Ukuran": status,
                "BCS / Kondisi Tubuh": bcs_batch,
                "Skor Akurasi Input": accuracy_score_batch,
                "Kategori Akurasi": accuracy_category_batch,
                "Kelas Pasar": kelas_pasar_batch,
                "Multiplier Kelas": kelas_multiplier_batch,
                "Margin Error (%)": margin_error_batch,
                "Harga per Kg": harga_hidup,
                "Harga per Kg Karkas": harga_karkas_batch,
                "Harga per Kg Daging": harga_daging_batch,
                "Estimasi Nilai Ternak": round(nilai_ternak, 0),
                "Estimasi Nilai Karkas": round(nilai_karkas, 0),
                "Estimasi Nilai Daging": round(nilai_daging, 0),
                "Harga Beli / Modal": round(harga_beli_modal_batch, 0),
                "Biaya Pemeliharaan": round(business["total_biaya_pemeliharaan"], 0),
                "Total Modal": round(business["total_modal"], 0),
                "Estimasi Keuntungan": round(business["estimasi_keuntungan"], 0),
                "ROI (%)": round(business["roi_percent"], 2),
                "Status Proses": "Berhasil",
            })
        except Exception as exc:
            results.append({
                "No": idx + 1,
                "Jenis Ternak": row.get("Jenis Ternak", ""),
                "Bangsa Ternak": row.get("Bangsa Ternak", ""),
                "Jenis Kelamin": row.get("Jenis Kelamin", ""),
                "Lingkar Dada (cm)": row.get("Lingkar Dada", ""),
                "Panjang Badan (cm)": row.get("Panjang Badan", ""),
                "Rumus": "-",
                "Prediksi Berat (kg)": 0,
                "BB Min (kg)": 0,
                "BB Max (kg)": 0,
                "Berat Karkas (kg)": 0,
                "Berat Daging (kg)": 0,
                "Status Ukuran": "-",
                "BCS / Kondisi Tubuh": row.get("BCS / Kondisi Tubuh", "Tidak dinilai"),
                "Skor Akurasi Input": 0,
                "Kategori Akurasi": "-",
                "Kelas Pasar": row.get("Kelas Pasar", "Otomatis"),
                "Multiplier Kelas": 0,
                "Margin Error (%)": row.get("Margin Error (%)", 10),
                "Harga per Kg": row.get("Harga per Kg", 0),
                "Harga per Kg Karkas": row.get("Harga per Kg Karkas", 0),
                "Harga per Kg Daging": row.get("Harga per Kg Daging", 0),
                "Estimasi Nilai Ternak": 0,
                "Estimasi Nilai Karkas": 0,
                "Estimasi Nilai Daging": 0,
                "Harga Beli / Modal": row.get("Harga Beli / Modal", 0),
                "Biaya Pemeliharaan": 0,
                "Total Modal": 0,
                "Estimasi Keuntungan": 0,
                "ROI (%)": 0,
                "Status Proses": f"Gagal: {exc}",
            })

    return pd.DataFrame(results)


# Judul dan deskripsi aplikasi
st.markdown("""
<div class="app-hero">
    <div class="app-pill">Prediksi Bobot • Karkas • Jagal • Blantik</div>
    <div class="app-hero-title">🐄 Prediksi Berat Badan Ternak</div>
    <p class="app-hero-subtitle">
        Hitung estimasi bobot hidup berdasarkan lingkar dada dan panjang badan, lalu baca hasilnya dari sudut pandang peternak, jagal, dan blantik berdasarkan jenis serta bangsa ternak.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="workflow-grid">
    <div class="workflow-step"><b>1. Input</b><span>Pilih jenis, bangsa, kelamin, LD, dan PB di sidebar.</span></div>
    <div class="workflow-step"><b>2. Hitung</b><span>Klik tombol Hitung Berat Badan untuk membuat estimasi utama.</span></div>
    <div class="workflow-step"><b>3. Baca Hasil</b><span>Lihat bobot, rentang error, BCS, dan skor akurasi.</span></div>
    <div class="workflow-step"><b>4. Analisis</b><span>Gunakan tab ekonomi, jagal, dan blantik sesuai kebutuhan.</span></div>
    <div class="workflow-step"><b>5. Simpan</b><span>Unduh PDF, CSV riwayat, atau proses banyak ternak sekaligus.</span></div>
</div>
""", unsafe_allow_html=True)

with st.expander("📏 Panduan pengukuran dan catatan akurasi", expanded=False):
    # Tambahkan panduan pengukuran
    st.markdown("### Panduan Pengukuran Ternak")

    # Buat tab untuk berbagai panduan pengukuran
    guide_tab1, guide_tab2, guide_tab3 = st.tabs([
        "📏 Cara Mengukur Lingkar Dada", 
        "📏 Cara Mengukur Panjang Badan",
        "⚖️ Tips Lainnya"
    ])

    with guide_tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            #### Cara Mengukur Lingkar Dada
        
            Pengukuran lingkar dada ternak dilakukan dengan cara:
        
            1. **Posisikan ternak** pada permukaan yang datar dan pastikan ternak berdiri dengan keempat kaki sejajar
            2. **Gunakan pita ukur** yang cukup panjang (meteran kain)
            3. **Lingkarkan pita ukur** di belakang bahu ternak (tepat di belakang kaki depan)
            4. **Pastikan pita** berada tepat di belakang bahu dan di depan rusuk pertama
            5. **Tarik pita** hingga cukup erat tapi tidak terlalu ketat (jangan sampai kulit ternak terlipat)
            6. **Catat hasil pengukuran** dalam satuan sentimeter (cm)
        
            > **Catatan Penting**: Pengukuran sebaiknya dilakukan pada pagi hari sebelum ternak diberi makan untuk menghindari pengembangan perut yang dapat mempengaruhi hasil pengukuran. Selain itu, pastikan ternak dalam keadaan seimbang dan tidak terlalu gelisah.
            """)
        with col2:
            show_image_safe("assets/lingkar_dada.png", "Gambar panduan menggunakan file karkas.jpeg.", fallback_paths=["assets/lingkar_dada.png", "version/V3/assets/panjangbadan.png"])

    with guide_tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            #### Cara Mengukur Panjang Badan
        
            Pengukuran panjang badan ternak dilakukan dengan cara:
        
            1. **Posisikan ternak** pada permukaan yang datar dan pastikan ternak berdiri dengan keempat kaki sejajar
            2. **Gunakan pita ukur** atau meteran yang kaku/rigid
            3. **Ukur jarak** dari tonjolan bahu (*tuberculum humeri*) sampai ke tonjolan tulang duduk (*tuberculum ischiadicum*) (**Gunakan Panjang Badan Absolut**)
            4. **Pastikan pengukuran** dilakukan dalam garis lurus dan horizontal (sejajar dengan tanah)
            5. **Catat hasil pengukuran** dalam satuan sentimeter (cm)
        
            > **Catatan**: Untuk memudahkan, Anda dapat menggunakan dua tongkat yang ditempatkan tegak lurus di depan bahu dan belakang tulang duduk, lalu ukur jarak antara keduanya.
            """)
        with col2:
            show_image_safe("assets/panjang_badan.png", "Gambar panduan menggunakan file karkas.jpeg.", fallback_paths=["assets/panjang_badan.png", "panjangbadan.png"])

    with guide_tab3:
        st.markdown("""
        #### Tips Tambahan untuk Pengukuran Akurat
    
        1. **Waktu Pengukuran**: Usahakan mengukur pada waktu yang sama dalam sehari, idealnya di pagi hari sebelum pemberian pakan.
        2. **Kondisi Ternak**: Pastikan ternak dalam kondisi tenang dan tidak stres. Pengukuran pada ternak yang gelisah bisa menghasilkan data yang tidak akurat.
        3. **Pengulangan**: Lakukan pengukuran 2-3 kali dan ambil nilai rata-rata untuk hasil yang lebih akurat.  
        4. **Pengukur**: Sebaiknya pengukuran dilakukan oleh orang yang sama jika ingin membandingkan hasil dari waktu ke waktu.   
        5. **Titik Referensi**: Gunakan titik-titik anatomi yang jelas dan konsisten sebagai referensi pengukuran.   
        6. **Ketelitian Pita Ukur**: Gunakan pita ukur yang tidak elastis dan pastikan pita tidak terlipat saat pengukuran.
        7. **Pencatatan**: Selalu catat tanggal pengukuran, karena pertumbuhan ternak dapat menyebabkan perubahan ukuran dalam periode waktu tertentu.
    
        ##### Perbandingan dengan Timbangan
    
        Meskipun metode pengukuran lingkar dada dan panjang badan merupakan pendekatan yang praktis untuk memprediksi berat badan ternak, 
        hasil prediksi ini tetap memiliki margin error sekitar 5-10% dibandingkan dengan penimbangan langsung menggunakan timbangan.
    
        Untuk keperluan yang membutuhkan keakuratan tinggi (seperti penjualan, kompetisi, atau penelitian), 
        sebaiknya tetap menggunakan timbangan ternak yang terkalibrasi dengan baik.
        """)

    # Tambahkan gambar panduan pengukuran
    # col1, col2 = st.columns([2,1]) # Removing this redundant image section
    # with col1:
    st.markdown(f"""
            =======================================
            > Aplikasi ini menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan 
            menggunakan **Rumus Formula** yang spesifik untuk jenis dan bangsa ternak yang berbeda. 
            Silakan pilih jenis dan bangsa ternak yang sesuai di sidebar untuk mendapatkan hasil yang lebih akurat.
        
            """)

    st.info(
        "Harga ekonomi memakai default acuan terbaru yang tersedia, tetapi tetap dapat diedit manual "
        "karena harga karkas dan daging berbeda antar daerah, kualitas potongan, dan waktu transaksi."
    )
    # with col2: # Removing this redundant image section
    # st.image("panjangbadan.png", caption="Panduan Pengukuran Panjang Badan, ref : https://vetmedicinae.com/cara-menghitung-berat-badan-sapi/", use_container_width=True)


# Sidebar untuk input pengguna
st.sidebar.header("1. Input Utama Ternak")

# Pilih jenis ternak
jenis_ternak = st.sidebar.selectbox(
    "Jenis Ternak",
    options=list(ANIMAL_DATA.keys()),
    help="Pilih jenis ternak yang ingin dihitung berat badannya."
)

# Pilih bangsa ternak
bangsa_ternak = st.sidebar.selectbox(
    "Bangsa Ternak",
    options=list(ANIMAL_DATA[jenis_ternak]["breeds"].keys()),
    help="Pilih bangsa ternak yang sesuai."
)

# Pilih jenis kelamin ternak
jenis_kelamin = st.sidebar.selectbox(
    "Jenis Kelamin",
    options=["Jantan", "Betina"],
    help="Pilih jenis kelamin ternak."
)

# Dapatkan rentang ukuran untuk bangsa ternak yang dipilih
breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
chest_range = breed_data["chest_range"]
length_range = breed_data["length_range"]

# Input lingkar dada dengan rentang sesuai bangsa ternak
lingkar_dada = st.sidebar.number_input(
    "4. Lingkar Dada (cm)",
    min_value=chest_range["min"] * 0.8,  # Sedikit di bawah minimum untuk fleksibilitas
    max_value=chest_range["max"] * 1.2,  # Sedikit di atas maksimum untuk fleksibilitas
    value=chest_range["min"] + (chest_range["max"] - chest_range["min"]) / 2,  # Nilai default di tengah rentang
    step=0.5,
    help=f"Ukur lingkar dada ternak dengan pita ukur, yaitu mengukur keliling dada ternak tepat di belakang bahu. Rentang normal untuk {bangsa_ternak}: {chest_range['min']}-{chest_range['max']} cm."
)

# Input panjang badan dengan rentang sesuai bangsa ternak
panjang_badan = st.sidebar.number_input(
    "5. Panjang Badan (cm)",
    min_value=length_range["min"] * 0.8,  # Sedikit di bawah minimum untuk fleksibilitas
    max_value=length_range["max"] * 1.2,  # Sedikit di atas maksimum untuk fleksibilitas
    value=length_range["min"] + (length_range["max"] - length_range["min"]) / 2,  # Nilai default di tengah rentang
    step=0.5,
    help=f"Ukur panjang badan ternak, yaitu dari ujung bahu hingga tulang duduk (tuber ischii). Rentang normal untuk {bangsa_ternak}: {length_range['min']}-{length_range['max']} cm."
)

# Nilai awal ekonomi/biaya dibuat default agar sidebar tetap fokus pada hitung berat badan.
# Input detail ekonomi dan biaya dipindahkan ke tab hasil setelah tombol Hitung Berat Badan ditekan.
status_preview, status_preview_note = get_size_status(
    lingkar_dada,
    panjang_badan,
    jenis_ternak,
    bangsa_ternak,
)

kelas_pasar_input = "Otomatis"
kelas_pasar, kelas_pasar_note, kelas_multiplier = get_market_class(
    status_preview,
    kelas_pasar_input,
)

latest_base_prices = get_latest_price_defaults(jenis_ternak, bangsa_ternak)
latest_prices = apply_market_class_to_prices(
    latest_base_prices,
    kelas_multiplier,
    kelas_pasar,
)

price_key_suffix = (
    f"{jenis_ternak}_{bangsa_ternak}_{kelas_pasar}"
    .replace(" ", "_")
    .replace("/", "_")
    .replace("(", "")
    .replace(")", "")
)

margin_error = 10
harga_bobot_hidup = int(latest_prices["harga_bobot_hidup"])
harga_karkas = int(latest_prices["harga_karkas"])
harga_daging = int(latest_prices["harga_daging"])
harga_beli_modal = 0
biaya_pakan_per_hari = 0
lama_pemeliharaan_hari = 0
biaya_obat_vitamin = 0
biaya_transportasi = 0
biaya_lain_lain = 0

# Tombol untuk menghitung berat badan
# st.session_state digunakan agar hasil tidak hilang ketika slider/komponen lain berubah.
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "calculation_history" not in st.session_state:
    st.session_state.calculation_history = []
if "new_calculation" not in st.session_state:
    st.session_state.new_calculation = False

st.sidebar.markdown("---")
st.sidebar.caption("Setelah data utama benar, klik tombol berikut untuk menampilkan hasil dan analisis.")
if st.sidebar.button("🚀 Hitung Berat Badan", type="primary"):
    st.session_state.show_results = True
    st.session_state.new_calculation = True

if st.session_state.show_results:
    # Add info message to guide users
    st.sidebar.info("👉 Alur baca hasil: Berat & Akurasi → Target → Ekonomi → Jagal → Blantik → Insight → Arsip.")
    
    # Hitung berat badan
    berat_badan, formula_name, formula_text = hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)

    if not validate_weight_result(berat_badan, jenis_ternak):
        st.warning(
            "Hasil prediksi terlihat berada di luar rentang wajar. "
            "Periksa kembali satuan pengukuran, rumus yang digunakan, dan data input."
        )
    
    # Hitung komponen karkas
    karkas_data = hitung_komponen_karkas(berat_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)
    
    # Area hasil dibuat bertab agar fokus utama tetap pada hitung berat badan.
    status_ukuran, status_note = get_size_status(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak)

    st.markdown("""
    <div class="section-note">
        <b>Urutan membaca hasil:</b> mulai dari tab 1 untuk bobot utama, lanjut ke target berat, ekonomi ternak, biaya/profit, jagal, blantik, lalu insight ringkas.
    </div>
    """, unsafe_allow_html=True)

    hasil_tab, target_tab, ekonomi_tab, biaya_tab, jagal_tab, blantik_tab, insight_tab, prompt_tab = st.tabs([
        "1️⃣ Berat & Akurasi",
        "2️⃣ Target Berat",
        "3️⃣ Ekonomi Ternak",
        "4️⃣ Biaya & Profit",
        "5️⃣ Jagal",
        "6️⃣ Blantik",
        "7️⃣ Insight",
        "8️⃣ Prompt AI",
    ])

    with hasil_tab:
        st.success(f"## Prediksi Berat Badan: **{berat_badan:.2f} kg**")

        margin_error = st.slider(
            "Margin error prediksi (%)",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
            key="margin_error_hasil_tab",
            help="Rentang estimasi bawah dan atas untuk menghindari hasil dianggap sebagai angka pasti."
        )

        bcs_option = st.selectbox(
            "BCS / Kondisi Tubuh",
            options=BCS_OPTIONS,
            index=0,
            key="bcs_hasil_tab",
            help="BCS membantu membaca kualitas estimasi. Pilih Tidak dinilai jika belum melakukan penilaian kondisi tubuh."
        )

        bb_min, bb_max = calculate_error_range(berat_badan, margin_error)
        karkas_min, karkas_max = calculate_error_range(karkas_data["karkas_weight"], margin_error)
        daging_min, daging_max = calculate_error_range(karkas_data["meat_weight"], margin_error)

        accuracy_score, accuracy_category, accuracy_notes = calculate_input_accuracy_score(
            lingkar_dada,
            panjang_badan,
            berat_badan,
            jenis_ternak,
            bangsa_ternak,
            bcs_option,
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Prediksi Berat", f"{berat_badan:.2f} kg")
            st.caption(f"Rentang ±{margin_error}%: {bb_min:.2f}–{bb_max:.2f} kg")
        with summary_col2:
            st.metric("Status Ukuran", status_ukuran)
            st.caption(status_note)
        with summary_col3:
            st.metric("Skor Akurasi Input", f"{accuracy_score}/100")
            st.caption(accuracy_category)
        with summary_col4:
            st.metric("Rumus", formula_name)
            st.caption(formula_text)

        with st.expander("Lihat catatan skor akurasi dan BCS"):
            st.write(f"**BCS/Kondisi tubuh:** {bcs_option}")
            st.write(BCS_NOTES.get(bcs_option, ""))
            for note in accuracy_notes:
                st.write(f"- {note}")

        st.markdown("#### Input Utama")
        input_summary_df = pd.DataFrame([
            {"Parameter": "Jenis Ternak", "Nilai": jenis_ternak},
            {"Parameter": "Bangsa Ternak", "Nilai": bangsa_ternak},
            {"Parameter": "Jenis Kelamin", "Nilai": jenis_kelamin},
            {"Parameter": "Lingkar Dada", "Nilai": f"{lingkar_dada:.1f} cm"},
            {"Parameter": "Panjang Badan", "Nilai": f"{panjang_badan:.1f} cm"},
            {"Parameter": "Rentang Normal LD", "Nilai": f"{chest_range['min']}–{chest_range['max']} cm"},
            {"Parameter": "Rentang Normal PB", "Nilai": f"{length_range['min']}–{length_range['max']} cm"},
        ])
        st.dataframe(input_summary_df, use_container_width=True, hide_index=True)

        with st.expander("Lihat profil jenis dan bangsa ternak"):
            breed_profile_df = create_breed_perspective_dataframe(jenis_ternak, bangsa_ternak)
            st.dataframe(breed_profile_df, use_container_width=True, hide_index=True)

        st.info(
            f"Fokus utama aplikasi adalah estimasi berat badan. "
            f"Hasil saat ini berada pada rentang **{bb_min:.2f}–{bb_max:.2f} kg** dengan margin error ±{margin_error}%. "
            f"Profil {bangsa_ternak}: {get_breed_business_profile(jenis_ternak, bangsa_ternak)['market_position']}"
        )

    with target_tab:
        st.markdown("#### Simulasi Target Berat")
        st.write(
            "Gunakan fitur ini untuk memperkirakan lingkar dada dan panjang badan yang diperlukan "
            "agar ternak mendekati target berat tertentu. Contoh target sudah disesuaikan dengan "
            "**jenis ternak + bangsa ternak** yang dipilih."
        )

        breed_target_examples = get_breed_target_examples(
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )

        st.markdown("#### Contoh Target Berdasarkan Jenis dan Bangsa")
        st.dataframe(breed_target_examples, use_container_width=True, hide_index=True)

        target_option_labels = create_breed_target_option_labels(breed_target_examples)
        recommended_target = get_recommended_target_weight_by_breed(
            berat_badan,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )

        selected_target_label = st.selectbox(
            "Pilih contoh target",
            options=target_option_labels,
            index=min(
                len(target_option_labels) - 1,
                max(
                    0,
                    next(
                        (
                            i for i, label in enumerate(target_option_labels)
                            if float(label.split(" - ")[1].replace(" kg", "")) >= recommended_target
                        ),
                        1,
                    )
                )
            ),
            key=f"target_option_{jenis_ternak}_{bangsa_ternak}_{jenis_kelamin}",
            help="Contoh target dihitung dari rentang normal lingkar dada dan panjang badan bangsa ternak."
        )

        selected_target_weight = float(
            selected_target_label.split(" - ")[1].replace(" kg", "")
        )

        target_weight = st.number_input(
            "Target Berat Badan (kg)",
            min_value=0.0,
            value=float(round(selected_target_weight, 2)),
            step=5.0 if jenis_ternak == "Sapi" else 1.0,
            key=f"target_berat_tab_{jenis_ternak}_{bangsa_ternak}_{jenis_kelamin}_{selected_target_label}",
            help="Anda tetap bisa mengubah angka target secara manual."
        )

        target_estimate = estimate_dimensions_for_target_weight(
            target_weight,
            lingkar_dada,
            panjang_badan,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )

        target_col1, target_col2, target_col3, target_col4 = st.columns(4)
        with target_col1:
            st.metric("Target Berat", f"{target_weight:.2f} kg")
        with target_col2:
            st.metric("Estimasi LD", f"{target_estimate['lingkar_dada']:.1f} cm")
        with target_col3:
            st.metric("Estimasi PB", f"{target_estimate['panjang_badan']:.1f} cm")
        with target_col4:
            st.metric("Status Target", target_estimate["status"])

        st.info(target_estimate["note"])
        st.caption(get_breed_specific_target_note(jenis_ternak, bangsa_ternak, target_estimate["status"]))

        target_compare_df = pd.DataFrame([
            {"Parameter": "Berat Saat Ini", "Nilai": f"{berat_badan:.2f} kg"},
            {"Parameter": "Target Berat", "Nilai": f"{target_weight:.2f} kg"},
            {"Parameter": "Selisih Berat", "Nilai": f"{target_weight - berat_badan:.2f} kg"},
            {"Parameter": "LD Saat Ini", "Nilai": f"{lingkar_dada:.1f} cm"},
            {"Parameter": "Estimasi LD Target", "Nilai": f"{target_estimate['lingkar_dada']:.1f} cm"},
            {"Parameter": "PB Saat Ini", "Nilai": f"{panjang_badan:.1f} cm"},
            {"Parameter": "Estimasi PB Target", "Nilai": f"{target_estimate['panjang_badan']:.1f} cm"},
        ])
        st.dataframe(target_compare_df, use_container_width=True, hide_index=True)

        st.markdown("#### Tabel Simulasi Target")
        target_table = create_target_simulation_table(
            berat_badan,
            lingkar_dada,
            panjang_badan,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
        )
        st.dataframe(target_table, use_container_width=True, hide_index=True)

        st.caption(
            "Catatan: simulasi ini adalah pendekatan matematis dari rumus yang digunakan aplikasi, "
            "bukan prediksi pertumbuhan biologis. Faktor pakan, umur, kesehatan, dan genetik tetap berpengaruh."
        )


    with ekonomi_tab:
        st.markdown("#### Estimasi Ekonomi")
        status_preview, _ = get_size_status(
            lingkar_dada,
            panjang_badan,
            jenis_ternak,
            bangsa_ternak,
        )

        kelas_pasar_input = st.selectbox(
            "Kelas/Kondisi Pasar",
            options=MARKET_CLASS_OPTIONS,
            key="kelas_pasar_ekonomi_tab",
            help="Pilih otomatis agar aplikasi menilai kelas dari ukuran tubuh, atau pilih manual sesuai kondisi lapangan."
        )

        kelas_pasar, kelas_pasar_note, kelas_multiplier = get_market_class(
            status_preview,
            kelas_pasar_input,
        )

        latest_base_prices = get_latest_price_defaults(jenis_ternak, bangsa_ternak)
        latest_prices = apply_market_class_to_prices(
            latest_base_prices,
            kelas_multiplier,
            kelas_pasar,
        )

        st.caption(latest_prices["label"])
        st.caption(f"Sumber/acuan: {latest_prices['source']}")
        st.info(get_breed_specific_price_note(jenis_ternak, bangsa_ternak))

        price_key_suffix = (
            f"{jenis_ternak}_{bangsa_ternak}_{kelas_pasar}"
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

        price_col1, price_col2, price_col3 = st.columns(3)
        with price_col1:
            harga_bobot_hidup = st.number_input(
                "Harga per kg bobot hidup (Rp)",
                min_value=0,
                value=int(latest_prices["harga_bobot_hidup"]),
                step=1000,
                key=f"harga_bobot_hidup_tab_{price_key_suffix}",
                help="Harga default mengikuti jenis, bangsa, dan kelas pasar ternak. Ubah manual jika harga daerah berbeda."
            )
        with price_col2:
            harga_karkas = st.number_input(
                "Harga per kg karkas (Rp)",
                min_value=0,
                value=int(latest_prices["harga_karkas"]),
                step=1000,
                key=f"harga_karkas_tab_{price_key_suffix}",
                help="Harga default mengikuti jenis, bangsa, dan kelas pasar ternak. Ubah manual jika harga daerah berbeda."
            )
        with price_col3:
            harga_daging = st.number_input(
                "Harga per kg daging (Rp)",
                min_value=0,
                value=int(latest_prices["harga_daging"]),
                step=1000,
                key=f"harga_daging_tab_{price_key_suffix}",
                help="Harga default mengikuti jenis, bangsa, dan kelas pasar ternak. Ubah manual jika harga daerah berbeda."
            )

        nilai_hidup = berat_badan * harga_bobot_hidup
        nilai_karkas = karkas_data["karkas_weight"] * harga_karkas
        nilai_daging = karkas_data["meat_weight"] * harga_daging
        nilai_hidup_min, nilai_hidup_max = calculate_error_range(nilai_hidup, margin_error)

        econ_col1, econ_col2, econ_col3 = st.columns(3)
        with econ_col1:
            st.metric("Nilai Bobot Hidup", format_rupiah(nilai_hidup))
            st.caption(f"Harga/kg: {format_rupiah(harga_bobot_hidup)} | Rentang: {format_rupiah(nilai_hidup_min)}–{format_rupiah(nilai_hidup_max)}")
        with econ_col2:
            st.metric("Nilai Karkas", format_rupiah(nilai_karkas))
            st.caption(f"Harga/kg: {format_rupiah(harga_karkas)}")
        with econ_col3:
            st.metric("Nilai Daging", format_rupiah(nilai_daging))
            st.caption(f"Harga/kg: {format_rupiah(harga_daging)}")

        st.info(f"**Kelas pasar:** {kelas_pasar}. {kelas_pasar_note}")

    with biaya_tab:
        st.markdown("#### Biaya & Keuntungan")

        cost_col1, cost_col2, cost_col3 = st.columns(3)
        with cost_col1:
            harga_beli_modal = st.number_input(
                "Harga beli/modal awal (Rp)",
                min_value=0,
                value=0,
                step=100000,
                key="harga_beli_modal_biaya_tab",
                help="Isi jika ingin menghitung estimasi keuntungan. Biarkan 0 jika tidak ada data modal awal."
            )
            biaya_obat_vitamin = st.number_input(
                "Biaya obat/vitamin (Rp)",
                min_value=0,
                value=0,
                step=10000,
                key="biaya_obat_biaya_tab",
            )
        with cost_col2:
            biaya_pakan_per_hari = st.number_input(
                "Biaya pakan per hari (Rp)",
                min_value=0,
                value=0,
                step=5000,
                key="biaya_pakan_biaya_tab",
            )
            biaya_transportasi = st.number_input(
                "Biaya transportasi (Rp)",
                min_value=0,
                value=0,
                step=10000,
                key="biaya_transport_biaya_tab",
            )
        with cost_col3:
            lama_pemeliharaan_hari = st.number_input(
                "Lama pemeliharaan (hari)",
                min_value=0,
                value=0,
                step=1,
                key="lama_pemeliharaan_biaya_tab",
            )
            biaya_lain_lain = st.number_input(
                "Biaya lain-lain (Rp)",
                min_value=0,
                value=0,
                step=10000,
                key="biaya_lain_biaya_tab",
            )

        business_metrics = calculate_maintenance_metrics(
            nilai_jual=nilai_hidup,
            harga_beli_modal=harga_beli_modal,
            biaya_pakan_per_hari=biaya_pakan_per_hari,
            lama_pemeliharaan_hari=lama_pemeliharaan_hari,
            biaya_obat_vitamin=biaya_obat_vitamin,
            biaya_transportasi=biaya_transportasi,
            biaya_lain_lain=biaya_lain_lain,
        )

        profit_col1, profit_col2, profit_col3 = st.columns(3)
        with profit_col1:
            st.metric("Total Biaya Pemeliharaan", format_rupiah(business_metrics["total_biaya_pemeliharaan"]))
        with profit_col2:
            st.metric("Total Modal", format_rupiah(business_metrics["total_modal"]))
        with profit_col3:
            st.metric("Estimasi Keuntungan", format_rupiah(business_metrics["estimasi_keuntungan"]))
            st.caption(f"ROI: {business_metrics['roi_percent']:.1f}%")

        biaya_df = pd.DataFrame([
            {"Komponen": "Harga beli/modal awal", "Nilai": harga_beli_modal},
            {"Komponen": "Biaya pakan total", "Nilai": business_metrics["biaya_pakan_total"]},
            {"Komponen": "Biaya obat/vitamin", "Nilai": biaya_obat_vitamin},
            {"Komponen": "Biaya transportasi", "Nilai": biaya_transportasi},
            {"Komponen": "Biaya lain-lain", "Nilai": biaya_lain_lain},
            {"Komponen": "Total biaya pemeliharaan", "Nilai": business_metrics["total_biaya_pemeliharaan"]},
            {"Komponen": "Total modal", "Nilai": business_metrics["total_modal"]},
            {"Komponen": "Estimasi keuntungan", "Nilai": business_metrics["estimasi_keuntungan"]},
        ])
        biaya_df["Nilai"] = biaya_df["Nilai"].apply(format_rupiah)
        st.dataframe(biaya_df, use_container_width=True, hide_index=True)

        if business_metrics["estimasi_keuntungan"] < 0:
            st.warning("Estimasi keuntungan masih negatif. Cek kembali harga beli/modal, biaya pakan, atau harga jual.")
        elif business_metrics["total_modal"] > 0:
            st.success("Estimasi keuntungan positif berdasarkan data biaya dan harga jual yang dimasukkan.")


    with jagal_tab:
        st.markdown("#### Kalkulator Jagal")
        st.write(
            "Fitur ini membantu memperkirakan omzet hasil potong, total modal, keuntungan, ROI, "
            "harga beli impas, dan harga beli maksimal agar tidak melewati target margin."
        )

        jagal_profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)
        st.info(f"Sudut pandang jagal untuk {bangsa_ternak}: {jagal_profile['butcher_view']}")

        default_harga_beli_jagal = int(round(nilai_hidup, 0)) if "nilai_hidup" in locals() else 0
        default_harga_daging_jagal = int(harga_daging) if "harga_daging" in locals() else int(latest_prices["harga_daging"])

        jagal_col1, jagal_col2, jagal_col3 = st.columns(3)
        with jagal_col1:
            harga_beli_ternak_jagal = st.number_input(
                "Harga beli ternak (Rp)",
                min_value=0,
                value=default_harga_beli_jagal,
                step=100000,
                key="harga_beli_ternak_jagal",
                help="Harga beli aktual dari ternak hidup."
            )
            harga_jual_daging_jagal = st.number_input(
                "Harga jual daging/kg (Rp)",
                min_value=0,
                value=default_harga_daging_jagal,
                step=1000,
                key="harga_jual_daging_jagal",
            )
            biaya_pemotongan_jagal = st.number_input(
                "Biaya pemotongan (Rp)",
                min_value=0,
                value=0,
                step=50000,
                key="biaya_pemotongan_jagal",
            )
        with jagal_col2:
            harga_jual_tulang_lemak_jagal = st.number_input(
                "Harga jual tulang & lemak/kg (Rp)",
                min_value=0,
                value=30000,
                step=1000,
                key="harga_jual_tulang_lemak_jagal",
            )
            biaya_transportasi_jagal = st.number_input(
                "Biaya transportasi (Rp)",
                min_value=0,
                value=0,
                step=50000,
                key="biaya_transportasi_jagal",
            )
            biaya_tenaga_kerja_jagal = st.number_input(
                "Biaya tenaga kerja (Rp)",
                min_value=0,
                value=0,
                step=50000,
                key="biaya_tenaga_kerja_jagal",
            )
        with jagal_col3:
            harga_jual_non_karkas_jagal = st.number_input(
                "Harga non-karkas rata-rata/kg (Rp)",
                min_value=0,
                value=25000,
                step=1000,
                key="harga_jual_non_karkas_jagal",
                help="Dipakai sebagai pendekatan untuk kepala, kulit, kaki, ekor, jeroan, darah, dan komponen non-karkas lain."
            )
            biaya_es_penyimpanan_jagal = st.number_input(
                "Biaya es/penyimpanan (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_es_penyimpanan_jagal",
            )
            biaya_sewa_retribusi_jagal = st.number_input(
                "Biaya sewa/retribusi (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_sewa_retribusi_jagal",
            )

        extra_col1, extra_col2 = st.columns(2)
        with extra_col1:
            biaya_lain_lain_jagal = st.number_input(
                "Biaya lain-lain jagal (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_lain_lain_jagal",
            )
        with extra_col2:
            target_margin_jagal = st.slider(
                "Target margin jagal (%)",
                min_value=0,
                max_value=40,
                value=10,
                step=1,
                key="target_margin_jagal",
                help="Dipakai untuk menghitung harga beli maksimal agar target margin tercapai."
            )

        jagal_metrics = calculate_butcher_metrics(
            karkas_data=karkas_data,
            harga_beli_ternak=harga_beli_ternak_jagal,
            harga_jual_daging=harga_jual_daging_jagal,
            harga_jual_tulang_lemak=harga_jual_tulang_lemak_jagal,
            harga_jual_non_karkas=harga_jual_non_karkas_jagal,
            biaya_pemotongan=biaya_pemotongan_jagal,
            biaya_transportasi=biaya_transportasi_jagal,
            biaya_tenaga_kerja=biaya_tenaga_kerja_jagal,
            biaya_es_penyimpanan=biaya_es_penyimpanan_jagal,
            biaya_sewa_retribusi=biaya_sewa_retribusi_jagal,
            biaya_lain_lain=biaya_lain_lain_jagal,
            target_margin_percent=target_margin_jagal,
        )

        st.markdown("#### Ringkasan Keputusan Jagal")
        decision_col1, decision_col2, decision_col3, decision_col4 = st.columns(4)
        with decision_col1:
            st.metric("Omzet Estimasi", format_rupiah(jagal_metrics["omzet_total"]))
        with decision_col2:
            st.metric("Total Modal", format_rupiah(jagal_metrics["total_modal"]))
        with decision_col3:
            st.metric("Estimasi Profit", format_rupiah(jagal_metrics["profit"]))
            st.caption(f"ROI: {jagal_metrics['roi_percent']:.1f}%")
        with decision_col4:
            st.metric("Keputusan", jagal_metrics["decision"])

        if jagal_metrics["decision"] == "Layak Dibeli":
            st.success(jagal_metrics["decision_note"])
        elif jagal_metrics["decision"] == "Perlu Negosiasi":
            st.warning(jagal_metrics["decision_note"])
        else:
            st.error(jagal_metrics["decision_note"])

        st.markdown("#### Harga Beli Maksimal")
        buy_col1, buy_col2, buy_col3 = st.columns(3)
        with buy_col1:
            st.metric("Harga Beli Impas", format_rupiah(jagal_metrics["break_even_buy_price"]))
        with buy_col2:
            st.metric("Harga Beli Maks. Target Margin", format_rupiah(max(0, jagal_metrics["max_buy_price"])))
        with buy_col3:
            st.metric("Target Profit", format_rupiah(jagal_metrics["target_profit"]))

        st.markdown("#### Estimasi Hasil Potong")
        component_df = create_butcher_component_dataframe(
            karkas_data,
            harga_jual_daging_jagal,
            harga_jual_tulang_lemak_jagal,
            harga_jual_non_karkas_jagal,
        )
        st.dataframe(component_df, use_container_width=True, hide_index=True)

        st.caption(
            f"Kesesuaian jagal berdasarkan profil {bangsa_ternak}: "
            f"{jagal_profile.get('butcher_fit', 5)}/10. {jagal_profile.get('strategy', '')}"
        )

        with st.expander("Lihat detail non-karkas"):
            non_karkas_detail_df = create_non_karkas_detail_dataframe(
                karkas_data,
                harga_jual_non_karkas_jagal,
            )
            st.dataframe(non_karkas_detail_df, use_container_width=True, hide_index=True)

        st.markdown("#### Rincian Biaya Jagal")
        jagal_cost_df = pd.DataFrame([
            {"Komponen": "Harga beli ternak", "Nilai": harga_beli_ternak_jagal},
            {"Komponen": "Biaya pemotongan", "Nilai": biaya_pemotongan_jagal},
            {"Komponen": "Biaya transportasi", "Nilai": biaya_transportasi_jagal},
            {"Komponen": "Biaya tenaga kerja", "Nilai": biaya_tenaga_kerja_jagal},
            {"Komponen": "Biaya es/penyimpanan", "Nilai": biaya_es_penyimpanan_jagal},
            {"Komponen": "Biaya sewa/retribusi", "Nilai": biaya_sewa_retribusi_jagal},
            {"Komponen": "Biaya lain-lain", "Nilai": biaya_lain_lain_jagal},
            {"Komponen": "Biaya operasional", "Nilai": jagal_metrics["biaya_operasional"]},
            {"Komponen": "Total modal", "Nilai": jagal_metrics["total_modal"]},
        ])
        jagal_cost_df["Nilai"] = jagal_cost_df["Nilai"].apply(format_rupiah)
        st.dataframe(jagal_cost_df, use_container_width=True, hide_index=True)

        st.markdown("#### Rekomendasi Jagal")
        for recommendation in generate_butcher_recommendations(jagal_metrics):
            st.write(f"- {recommendation}")

        st.caption(
            "Catatan: nilai non-karkas memakai harga rata-rata gabungan. Untuk transaksi aktual, "
            "harga kulit, kepala, kaki, jeroan, dan komponen lain sebaiknya disesuaikan dengan pasar setempat."
        )



    with insight_tab:
        st.markdown("#### Insight Analisis Otomatis")
        st.write(
            "Tab ini merangkum hasil utama menjadi insight praktis untuk membaca kualitas ternak, "
            "risiko transaksi, dan prioritas tindakan."
        )

        insight_profile_df = create_breed_perspective_dataframe(jenis_ternak, bangsa_ternak)
        with st.expander("Profil sudut pandang berdasarkan jenis dan bangsa"):
            st.dataframe(insight_profile_df, use_container_width=True, hide_index=True)

        insights = calculate_operational_insights(
            berat_badan=berat_badan,
            karkas_data=karkas_data,
            accuracy_score=accuracy_score,
            bcs_option=bcs_option,
            jagal_metrics=jagal_metrics if "jagal_metrics" in locals() else None,
            jenis_ternak=jenis_ternak,
            bangsa_ternak=bangsa_ternak,
        )

        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        with insight_col1:
            st.metric("Efisiensi Karkas", f"{insights['karkas_yield']:.1f}%")
            st.caption(insights["karkas_category"])
        with insight_col2:
            st.metric("Daging / Bobot Hidup", f"{insights['meat_yield_live']:.1f}%")
            st.caption(insights["meat_category"])
        with insight_col3:
            st.metric("Daging / Karkas", f"{insights['meat_yield_carcass']:.1f}%")
            st.caption("Proporsi daging dari karkas")
        with insight_col4:
            st.metric("Non-Karkas / Bobot Hidup", f"{insights['non_karkas_ratio']:.1f}%")
            st.caption("Kepala, kulit, kaki, jeroan, dll.")

        st.markdown("#### Ringkasan Komposisi Hasil Potong")
        composition_df = pd.DataFrame([
            {"Indikator": "Bobot hidup", "Nilai": f"{berat_badan:.2f} kg", "Insight": "Dasar estimasi seluruh hasil potong"},
            {"Indikator": "Berat karkas", "Nilai": f"{karkas_data['karkas_weight']:.2f} kg", "Insight": f"Efisiensi {insights['karkas_yield']:.1f}%"},
            {"Indikator": "Berat daging bersih", "Nilai": f"{karkas_data['meat_weight']:.2f} kg", "Insight": f"{insights['meat_yield_live']:.1f}% dari bobot hidup"},
            {"Indikator": "Tulang & lemak karkas", "Nilai": f"{karkas_data['bone_and_fat_weight']:.2f} kg", "Insight": f"{insights['bone_fat_ratio']:.1f}% dari karkas"},
            {"Indikator": "Non-karkas", "Nilai": f"{sum(karkas_data['non_karkas_weights'].values()):.2f} kg", "Insight": f"{insights['non_karkas_ratio']:.1f}% dari bobot hidup"},
        ])
        st.dataframe(composition_df, use_container_width=True, hide_index=True)

        st.markdown("#### Risiko Utama")
        for note in insights["risk_notes"]:
            st.warning(note)

        st.markdown("#### Peluang / Sisi Positif")
        for note in insights["opportunity_notes"]:
            st.success(note)

        if "jagal_metrics" in locals():
            st.markdown("#### Sensitivitas Harga Jual")
            price_sensitivity_df = create_price_sensitivity_dataframe(
                karkas_data=karkas_data,
                harga_jual_daging=harga_jual_daging_jagal,
                harga_jual_tulang_lemak=harga_jual_tulang_lemak_jagal,
                harga_jual_non_karkas=harga_jual_non_karkas_jagal,
                harga_beli_ternak=harga_beli_ternak_jagal,
                biaya_operasional=jagal_metrics["biaya_operasional"],
            )
            st.dataframe(price_sensitivity_df, use_container_width=True, hide_index=True)

            st.markdown("#### Sensitivitas Susut Daging")
            shrinkage_df = create_shrinkage_sensitivity_dataframe(
                jagal_metrics=jagal_metrics,
                harga_jual_daging=harga_jual_daging_jagal,
            )
            st.dataframe(shrinkage_df, use_container_width=True, hide_index=True)

            st.markdown("#### Struktur Omzet Jagal")
            omzet_total = jagal_metrics["omzet_total"] if jagal_metrics["omzet_total"] > 0 else 1
            omzet_structure_df = pd.DataFrame([
                {
                    "Sumber Omzet": "Daging",
                    "Nilai": format_rupiah(jagal_metrics["omzet_daging"]),
                    "Kontribusi": f"{(jagal_metrics['omzet_daging'] / omzet_total) * 100:.1f}%",
                },
                {
                    "Sumber Omzet": "Tulang & lemak",
                    "Nilai": format_rupiah(jagal_metrics["omzet_tulang_lemak"]),
                    "Kontribusi": f"{(jagal_metrics['omzet_tulang_lemak'] / omzet_total) * 100:.1f}%",
                },
                {
                    "Sumber Omzet": "Non-karkas",
                    "Nilai": format_rupiah(jagal_metrics["omzet_non_karkas"]),
                    "Kontribusi": f"{(jagal_metrics['omzet_non_karkas'] / omzet_total) * 100:.1f}%",
                },
            ])
            st.dataframe(omzet_structure_df, use_container_width=True, hide_index=True)

        st.markdown("#### Checklist Keputusan")
        for item in create_decision_checklist(
            insights,
            jagal_metrics if "jagal_metrics" in locals() else None,
        ):
            st.write(f"- {item}")

        st.caption(
            "Insight ini membantu membaca hasil secara cepat. Keputusan akhir tetap perlu mempertimbangkan "
            "pemeriksaan fisik, timbangan aktual, harga pasar, dan biaya nyata di lapangan."
        )



    with blantik_tab:
        st.markdown("#### Insight Blantik Ternak")
        st.write(
            "Tab ini membantu membaca transaksi dari sudut pandang blantik: harga beli wajar, "
            "harga jual kembali, margin, batas nego, daya jual, segmentasi pembeli, dan strategi jual."
        )

        blantik_profile = get_breed_business_profile(jenis_ternak, bangsa_ternak)
        st.info(f"Sudut pandang blantik untuk {bangsa_ternak}: {blantik_profile['trader_view']}")

        default_harga_beli_blantik = int(round(nilai_hidup, 0)) if "nilai_hidup" in locals() else int(round(berat_badan * latest_prices["harga_bobot_hidup"], 0))
        default_harga_jual_blantik = int(harga_bobot_hidup) if "harga_bobot_hidup" in locals() else int(latest_prices["harga_bobot_hidup"])

        blantik_col1, blantik_col2, blantik_col3 = st.columns(3)
        with blantik_col1:
            harga_beli_blantik = st.number_input(
                "Harga beli ternak (Rp)",
                min_value=0,
                value=default_harga_beli_blantik,
                step=100000,
                key="harga_beli_blantik",
            )
            biaya_angkut_blantik = st.number_input(
                "Biaya angkut (Rp)",
                min_value=0,
                value=0,
                step=50000,
                key="biaya_angkut_blantik",
            )
            biaya_kandang_blantik = st.number_input(
                "Biaya kandang/pasar (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_kandang_blantik",
            )
        with blantik_col2:
            harga_jual_per_kg_blantik = st.number_input(
                "Estimasi harga jual/kg bobot hidup (Rp)",
                min_value=0,
                value=default_harga_jual_blantik,
                step=1000,
                key="harga_jual_per_kg_blantik",
            )
            biaya_pakan_harian_blantik = st.number_input(
                "Biaya pakan harian (Rp)",
                min_value=0,
                value=0,
                step=5000,
                key="biaya_pakan_harian_blantik",
            )
            biaya_retribusi_blantik = st.number_input(
                "Biaya retribusi/pasar (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_retribusi_blantik",
            )
        with blantik_col3:
            target_margin_blantik = st.slider(
                "Target margin blantik (%)",
                min_value=0,
                max_value=40,
                value=10,
                step=1,
                key="target_margin_blantik",
            )
            lama_tahan_hari_blantik = st.number_input(
                "Lama tahan sebelum jual (hari)",
                min_value=0,
                value=0,
                step=1,
                key="lama_tahan_hari_blantik",
            )
            biaya_tenaga_bantu_blantik = st.number_input(
                "Biaya tenaga bantu (Rp)",
                min_value=0,
                value=0,
                step=25000,
                key="biaya_tenaga_bantu_blantik",
            )

        biaya_lain_blantik = st.number_input(
            "Biaya lain-lain blantik (Rp)",
            min_value=0,
            value=0,
            step=25000,
            key="biaya_lain_blantik",
        )

        trader_insights = calculate_trader_insights(
            berat_badan=berat_badan,
            jenis_ternak=jenis_ternak,
            bangsa_ternak=bangsa_ternak,
            status_ukuran=status_ukuran,
            bcs_option=bcs_option,
            accuracy_score=accuracy_score,
            harga_beli=harga_beli_blantik,
            harga_jual_per_kg=harga_jual_per_kg_blantik,
            biaya_angkut=biaya_angkut_blantik,
            biaya_pakan_harian=biaya_pakan_harian_blantik,
            lama_tahan_hari=lama_tahan_hari_blantik,
            biaya_kandang=biaya_kandang_blantik,
            biaya_retribusi=biaya_retribusi_blantik,
            biaya_tenaga_bantu=biaya_tenaga_bantu_blantik,
            biaya_lain=biaya_lain_blantik,
            target_margin_percent=target_margin_blantik,
        )

        st.markdown("#### Ringkasan Transaksi Blantik")
        trader_col1, trader_col2, trader_col3, trader_col4 = st.columns(4)
        with trader_col1:
            st.metric("Estimasi Harga Jual", format_rupiah(trader_insights["estimasi_harga_jual"]))
        with trader_col2:
            st.metric("Margin Bersih", format_rupiah(trader_insights["margin_bersih"]))
            st.caption(f"ROI: {trader_insights['roi']:.1f}%")
        with trader_col3:
            st.metric("Skor Daya Jual", f"{trader_insights['resale_score']}/100")
            st.caption(trader_insights["resale_category"])
        with trader_col4:
            st.metric("Keputusan", trader_insights["decision"])

        if trader_insights["decision"] == "Layak Dibeli":
            st.success(trader_insights["decision_note"])
        elif trader_insights["decision"] in ["Perlu Negosiasi", "Tahan Dulu"]:
            st.warning(trader_insights["decision_note"])
        else:
            st.error(trader_insights["decision_note"])

        st.markdown("#### Batas Harga Nego")
        nego_col1, nego_col2, nego_col3 = st.columns(3)
        with nego_col1:
            st.metric("Harga Ideal Beli", format_rupiah(max(0, trader_insights["harga_beli_ideal"])))
        with nego_col2:
            st.metric("Harga Maksimal Beli", format_rupiah(max(0, trader_insights["harga_beli_maksimal"])))
        with nego_col3:
            st.metric("Harga Impas", format_rupiah(max(0, trader_insights["harga_beli_impas"])))

        st.markdown("#### Strategi Jual")
        strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
        with strategy_col1:
            st.metric("Strategi", trader_insights["strategy"])
            st.caption(trader_insights["strategy_note"])
        with strategy_col2:
            st.metric("Risiko Transaksi", trader_insights["risk_level"])
        with strategy_col3:
            st.metric("Target Profit", format_rupiah(trader_insights["target_profit"]))

        st.markdown("#### Segmentasi Calon Pembeli")
        segment_text = ", ".join(trader_insights["buyer_segments"]) if trader_insights["buyer_segments"] else "Belum teridentifikasi"
        st.info(segment_text)
        st.caption(
            f"Likuiditas profil {bangsa_ternak}: {blantik_profile.get('liquidity_bonus', 5)}/10. "
            f"Strategi umum: {blantik_profile.get('strategy', '')}"
        )

        st.markdown("#### Rincian Modal Blantik")
        trader_cost_df = pd.DataFrame([
            {"Komponen": "Harga beli ternak", "Nilai": harga_beli_blantik},
            {"Komponen": "Biaya angkut", "Nilai": biaya_angkut_blantik},
            {"Komponen": "Biaya pakan total", "Nilai": trader_insights["biaya_pakan_total"]},
            {"Komponen": "Biaya kandang/pasar", "Nilai": biaya_kandang_blantik},
            {"Komponen": "Biaya retribusi/pasar", "Nilai": biaya_retribusi_blantik},
            {"Komponen": "Biaya tenaga bantu", "Nilai": biaya_tenaga_bantu_blantik},
            {"Komponen": "Biaya lain-lain", "Nilai": biaya_lain_blantik},
            {"Komponen": "Total biaya tambahan", "Nilai": trader_insights["total_biaya_tambahan"]},
            {"Komponen": "Total modal", "Nilai": trader_insights["total_modal"]},
        ])
        trader_cost_df["Nilai"] = trader_cost_df["Nilai"].apply(format_rupiah)
        st.dataframe(trader_cost_df, use_container_width=True, hide_index=True)

        st.markdown("#### Sensitivitas Harga Jual Ulang")
        trader_sensitivity_df = create_trader_sensitivity_dataframe(
            berat_badan=berat_badan,
            harga_beli=harga_beli_blantik,
            harga_jual_per_kg=harga_jual_per_kg_blantik,
            total_biaya_tambahan=trader_insights["total_biaya_tambahan"],
        )
        st.dataframe(trader_sensitivity_df, use_container_width=True, hide_index=True)

        st.markdown("#### Risiko Blantik")
        for risk_note in trader_insights["risk_notes"]:
            st.warning(risk_note)
        st.markdown("#### Catatan Khusus Bangsa")
        st.write(f"- Posisi pasar: {blantik_profile['market_position']}")
        st.write(f"- Risiko khusus: {'; '.join(blantik_profile.get('risks', []))}")

        st.markdown("#### Checklist Tindakan")
        for item in create_trader_checklist(trader_insights):
            st.write(f"- {item}")

        st.caption(
            "Catatan: insight blantik memakai estimasi bobot hidup dan harga jual ulang. "
            "Tetap cek fisik, umur, kesehatan, dan harga pasar setempat sebelum deal."
        )



    with prompt_tab:
        st.markdown("#### Generator Prompt untuk AI Lain")
        st.write(
            "Fitur ini membuat prompt otomatis dari seluruh hasil perhitungan. "
            "Peternak dapat menyalin prompt ini ke AI lain untuk mendapatkan analisis lanjutan yang lebih detail."
        )

        prompt_mode = st.selectbox(
            "Pilih sudut pandang prompt",
            options=["Peternak", "Jagal", "Blantik", "Analisis Lengkap"],
            index=3,
            key="prompt_mode_ai",
            help="Pilih jenis analisis yang ingin diminta ke AI lain."
        )

        prompt_col1, prompt_col2, prompt_col3, prompt_col4 = st.columns(4)
        with prompt_col1:
            st.metric("Jenis", jenis_ternak)
        with prompt_col2:
            st.metric("Bangsa", bangsa_ternak)
        with prompt_col3:
            st.metric("Berat", f"{berat_badan:.2f} kg")
        with prompt_col4:
            st.metric("Skor Input", f"{accuracy_score}/100")

        ai_prompt_text = build_ai_prompt_from_results(
            prompt_mode=prompt_mode,
            jenis_ternak=jenis_ternak,
            bangsa_ternak=bangsa_ternak,
            jenis_kelamin=jenis_kelamin,
            lingkar_dada=lingkar_dada,
            panjang_badan=panjang_badan,
            berat_badan=berat_badan,
            bb_min=bb_min,
            bb_max=bb_max,
            margin_error=margin_error,
            formula_name=formula_name,
            formula_text=formula_text,
            status_ukuran=status_ukuran,
            status_note=status_note,
            bcs_option=bcs_option,
            accuracy_score=accuracy_score,
            accuracy_category=accuracy_category,
            karkas_data=karkas_data,
            breed_profile=get_breed_business_profile(jenis_ternak, bangsa_ternak),
            target_table=target_table if "target_table" in locals() else None,
            harga_bobot_hidup=harga_bobot_hidup,
            harga_karkas=harga_karkas,
            harga_daging=harga_daging,
            nilai_hidup=nilai_hidup,
            nilai_karkas=nilai_karkas,
            nilai_daging=nilai_daging,
            business_metrics=business_metrics,
            jagal_metrics=jagal_metrics if "jagal_metrics" in locals() else {},
            trader_insights=trader_insights if "trader_insights" in locals() else {},
            insights=insights if "insights" in locals() else {},
        )

        st.markdown("#### Prompt Siap Salin")
        st.text_area(
            "Salin prompt di bawah ini, lalu tempel ke AI lain.",
            value=ai_prompt_text,
            height=520,
            key="generated_ai_prompt_text_area"
        )

        download_filename = (
            f"prompt_ai_{jenis_ternak}_{bangsa_ternak}_{prompt_mode}"
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            + ".txt"
        )

        st.download_button(
            label="⬇️ Download Prompt TXT",
            data=ai_prompt_text.encode("utf-8"),
            file_name=download_filename,
            mime="text/plain"
        )

        with st.expander("Cara menggunakan prompt ini"):
            st.write("- Pilih sudut pandang prompt sesuai kebutuhan.")
            st.write("- Salin semua teks pada kotak prompt.")
            st.write("- Tempel ke AI lain seperti ChatGPT, Gemini, Claude, atau AI lain.")
            st.write("- Tambahkan konteks lokal, misalnya harga pasar daerah, umur ternak, jenis pakan, dan tujuan transaksi.")
            st.write("- Minta AI tersebut membuat rencana lanjutan yang lebih spesifik.")

        with st.expander("Contoh tambahan instruksi yang bisa ditambahkan"):
            st.code(
                "Tambahkan analisis berdasarkan harga pasar di daerah saya. "
                "Saya berada di [nama daerah], harga sapi/kambing/domba saat ini sekitar [harga]. "
                "Buatkan saran tindakan 30 hari ke depan.",
                language="text"
            )


    # Simpan riwayat hanya saat tombol hitung baru ditekan
    if st.session_state.new_calculation:
        st.session_state.calculation_history.append({
            "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Jenis Ternak": jenis_ternak,
            "Bangsa Ternak": bangsa_ternak,
            "Jenis Kelamin": jenis_kelamin,
            "Lingkar Dada (cm)": lingkar_dada,
            "Panjang Badan (cm)": panjang_badan,
            "Prediksi Berat (kg)": round(berat_badan, 2),
            "Berat Karkas (kg)": round(karkas_data["karkas_weight"], 2),
            "Berat Daging (kg)": round(karkas_data["meat_weight"], 2),
            "Status Ukuran": status_ukuran,
            "BCS / Kondisi Tubuh": bcs_option,
            "Skor Akurasi Input": accuracy_score,
            "Kategori Akurasi": accuracy_category,
            "Kelas Pasar": kelas_pasar,
            "Margin Error (%)": margin_error,
            "BB Min (kg)": round(bb_min, 2),
            "BB Max (kg)": round(bb_max, 2),
            "Harga/kg Bobot Hidup": harga_bobot_hidup,
            "Harga/kg Karkas": harga_karkas,
            "Harga/kg Daging": harga_daging,
            "Estimasi Nilai Ternak": round(nilai_hidup, 0),
            "Estimasi Nilai Karkas": round(nilai_karkas, 0),
            "Estimasi Nilai Daging": round(nilai_daging, 0),
            "Total Biaya Pemeliharaan": round(business_metrics["total_biaya_pemeliharaan"], 0),
            "Total Modal": round(business_metrics["total_modal"], 0),
            "Estimasi Keuntungan": round(business_metrics["estimasi_keuntungan"], 0),
            "ROI (%)": round(business_metrics["roi_percent"], 2),
            "Omzet Jagal": round(jagal_metrics["omzet_total"], 0) if "jagal_metrics" in locals() else 0,
            "Profit Jagal": round(jagal_metrics["profit"], 0) if "jagal_metrics" in locals() else 0,
            "ROI Jagal (%)": round(jagal_metrics["roi_percent"], 2) if "jagal_metrics" in locals() else 0,
            "Keputusan Jagal": jagal_metrics["decision"] if "jagal_metrics" in locals() else "-",
            "Harga Beli Maksimal Jagal": round(max(0, jagal_metrics["max_buy_price"]), 0) if "jagal_metrics" in locals() else 0,
            "Estimasi Harga Jual Blantik": round(trader_insights["estimasi_harga_jual"], 0) if "trader_insights" in locals() else 0,
            "Margin Bersih Blantik": round(trader_insights["margin_bersih"], 0) if "trader_insights" in locals() else 0,
            "ROI Blantik (%)": round(trader_insights["roi"], 2) if "trader_insights" in locals() else 0,
            "Skor Daya Jual": trader_insights["resale_score"] if "trader_insights" in locals() else 0,
            "Kategori Daya Jual": trader_insights["resale_category"] if "trader_insights" in locals() else "-",
            "Strategi Blantik": trader_insights["strategy"] if "trader_insights" in locals() else "-",
            "Keputusan Blantik": trader_insights["decision"] if "trader_insights" in locals() else "-",
            "Harga Maksimal Beli Blantik": round(max(0, trader_insights["harga_beli_maksimal"]), 0) if "trader_insights" in locals() else 0,
            "Profil Pasar Bangsa": get_breed_business_profile(jenis_ternak, bangsa_ternak)["market_position"],
            "Segmen Pembeli Utama": ", ".join(get_breed_business_profile(jenis_ternak, bangsa_ternak)["primary_buyers"]),
        })
        st.session_state.new_calculation = False
    
    st.markdown("---")
    with st.expander("🧾 Detail teknis, hasil potong, visualisasi, dan laporan", expanded=False):
        # Tampilkan detail perhitungan
        st.subheader("Detail Perhitungan:")
    
        # Dapatkan referensi dari formula
        formula_reference = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["reference"]
    
        st.markdown(f"""
        - Jenis Ternak: **{jenis_ternak}**
        - Bangsa Ternak: **{bangsa_ternak}**
        - Jenis Kelamin: **{jenis_kelamin}**
        - Profil Pasar Bangsa: **{get_breed_business_profile(jenis_ternak, bangsa_ternak)['market_position']}**
        - BCS / Kondisi Tubuh: **{bcs_option}**
        - Skor Akurasi Input: **{accuracy_score}/100 ({accuracy_category})**
        - Kelas Pasar: **{kelas_pasar}** (penyesuaian harga x{kelas_multiplier:.2f})
        - Margin Error: **±{margin_error}%** (rentang BB: {bb_min:.2f}–{bb_max:.2f} kg)
        - Rumus yang Digunakan: **{formula_name}**
        - Formula: **{formula_text}**
        - Referensi: **{formula_reference}**
        - Lingkar Dada (LD): **{lingkar_dada} cm** (Rentang normal: {chest_range['min']}-{chest_range['max']} cm)
        - Panjang Badan (PB): **{panjang_badan} cm** (Rentang normal: {length_range['min']}-{length_range['max']} cm)
        - Berat Badan (BB) = **{berat_badan:.2f} kg**
        """)

        st.subheader("Rekomendasi Otomatis")
        for recommendation in generate_recommendations(
            berat_badan,
            lingkar_dada,
            panjang_badan,
            jenis_ternak,
            bangsa_ternak,
            jenis_kelamin,
            kelas_pasar=kelas_pasar,
            margin_error=margin_error,
            estimasi_keuntungan=business_metrics["estimasi_keuntungan"],
            bcs_option=bcs_option,
            accuracy_score=accuracy_score,
        ):
            st.write(f"- {recommendation}")

        report_data = {
            "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "jenis_ternak": jenis_ternak,
            "bangsa_ternak": bangsa_ternak,
            "jenis_kelamin": jenis_kelamin,
            "lingkar_dada": lingkar_dada,
            "panjang_badan": panjang_badan,
            "formula_name": formula_name,
            "berat_badan": berat_badan,
            "karkas_weight": karkas_data["karkas_weight"],
            "meat_weight": karkas_data["meat_weight"],
            "bone_and_fat_weight": karkas_data["bone_and_fat_weight"],
            "status_ukuran": status_ukuran,
            "bcs_option": bcs_option,
            "accuracy_score": accuracy_score,
            "accuracy_category": accuracy_category,
            "kelas_pasar": kelas_pasar,
            "margin_error": margin_error,
            "bb_min": bb_min,
            "bb_max": bb_max,
            "karkas_min": karkas_min,
            "karkas_max": karkas_max,
            "daging_min": daging_min,
            "daging_max": daging_max,
            "harga_hidup": harga_bobot_hidup,
            "nilai_hidup": nilai_hidup,
            "nilai_hidup_min": nilai_hidup_min,
            "nilai_hidup_max": nilai_hidup_max,
            "harga_karkas": harga_karkas,
            "nilai_karkas": nilai_karkas,
            "harga_daging": harga_daging,
            "nilai_daging": nilai_daging,
            "total_biaya_pemeliharaan": business_metrics["total_biaya_pemeliharaan"],
            "total_modal": business_metrics["total_modal"],
            "estimasi_keuntungan": business_metrics["estimasi_keuntungan"],
            "roi_percent": business_metrics["roi_percent"],
        }
        pdf_bytes = create_pdf_report(report_data)
        if pdf_bytes:
            st.download_button(
                label="📄 Download Laporan PDF",
                data=pdf_bytes,
                file_name=f"laporan_prediksi_{jenis_ternak.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Fitur PDF membutuhkan package reportlab. Pastikan requirements.txt berisi reportlab.")
    
        # Tampilkan prediksi karkas, non-karkas, dan daging
        st.subheader("Prediksi Hasil Pemotongan:")
    
        # Buat layout kolom
        col1, col2 = st.columns([1, 1])
    
        with col1:
            # Informasi karkas dan daging
            st.markdown(f"""
            #### Karkas
            - Persentase karkas: **{karkas_data['karkas_percent']:.1f}%**
            - Berat karkas: **{karkas_data['karkas_weight']:.2f} kg**
        
            #### Daging
            - Persentase daging dari karkas: **{karkas_data['meat_percent_of_carcass']:.1f}%**
            - Persentase daging dari berat hidup: **{karkas_data['meat_percent_of_body']:.1f}%**
            - Berat daging total: **{karkas_data['meat_weight']:.2f} kg**
            - Berat tulang dan lemak karkas: **{karkas_data['bone_and_fat_weight']:.2f} kg**
        
            > *Referensi data: {karkas_data['reference']}*
            """)
    
        with col2:
            # Informasi komponen non-karkas
            st.markdown("#### Komponen Non-Karkas")
        
            # Buat dataframe untuk komponen non-karkas
            non_karkas_df = pd.DataFrame({
                "Komponen": list(karkas_data["non_karkas_weights"].keys()),
                "Berat (kg)": [f"{w:.2f}" for w in karkas_data["non_karkas_weights"].values()],
                "Persentase (%)": [f"{(w / berat_badan) * 100:.1f}" if berat_badan > 0 else "0.0" for w in karkas_data["non_karkas_weights"].values()]
            })
        
            # Tampilkan tabel
            st.dataframe(non_karkas_df, hide_index=True)
    
        # Visualisasi proporsi karkas dan non-karkas
        st.subheader("Visualisasi Proporsi Pemotongan")
    
        # Buat data untuk pie chart
        labels = ["Daging", "Tulang & Lemak Karkas"]
        values = [karkas_data["meat_weight"], karkas_data["bone_and_fat_weight"]]
    
        # Tambahkan komponen non-karkas
        for component, weight in karkas_data["non_karkas_weights"].items():
            if weight > 0.01 * berat_badan:  # Tampilkan hanya komponen yang signifikan (>1%)
                labels.append(component)
                values.append(weight)
    
        # Buat 2 kolom untuk visualisasi yang berbeda
        viz_col1, viz_col2 = st.columns([1, 1])
    
        with viz_col1:
            # Buat pie chart dengan plotly
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                textinfo='label+percent',
                insidetextorientation='radial',
                pull=[0.1 if x == "Daging" else 0 for x in labels],
                marker_colors=px.colors.qualitative.Pastel
            )])
        
            fig.update_layout(
                title_text=f"Proporsi Komponen Pemotongan<br>{jenis_ternak} {bangsa_ternak} {jenis_kelamin}",
                annotations=[dict(text=f'Total: {berat_badan:.1f} kg', x=0.5, y=0.5, font_size=12, showarrow=False)]
            )
        
            st.plotly_chart(fig, use_container_width=True)
    
        with viz_col2:
            # Buat treemap dengan plotly
            fig = px.treemap(
                names=labels,
                parents=["" for _ in labels],
                values=values,
                color=values,
                color_continuous_scale='Viridis',
                title=f"Treemap Komponen Pemotongan<br>{jenis_ternak} {bangsa_ternak} {jenis_kelamin}"
            )
        
            fig.update_traces(textinfo="label+value+percent parent")
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        
            st.plotly_chart(fig, use_container_width=True)
    
        # Perbandingan dengan bangsa lain
        st.subheader("Perbandingan Hasil Karkas dengan Bangsa Lain")
    
        # Kumpulkan data karkas untuk semua bangsa dari jenis ternak yang sama
        breeds = SLAUGHTER_DATA[jenis_ternak]["breeds"]
        breed_names = []
        karkas_percents = []
        meat_percents = []
    
        for breed_name, breed_data in breeds.items():
            breed_names.append(breed_name)
            karkas_percents.append(breed_data["karkas_percent"][jenis_kelamin])
            meat_of_karkas = breed_data["meat_percent_of_carcass"]
            meat_of_live = (breed_data["karkas_percent"][jenis_kelamin] * meat_of_karkas) / 100
            meat_percents.append(meat_of_live)
    
        # Buat dataframe
        comparison_df = pd.DataFrame({
            "Bangsa": breed_names,
            "Persentase Karkas (%)": karkas_percents,
            "Persentase Daging dari Berat Hidup (%)": meat_percents
        })
    
        # Tambahkan kolom berat karkas dan daging untuk berat badan saat ini
        comparison_df["Berat Karkas (kg)"] = [(p * berat_badan) / 100 for p in karkas_percents]
        comparison_df["Berat Daging (kg)"] = [(p * berat_badan) / 100 for p in meat_percents]
    
        # Tampilkan tabel dengan highlight
        st.dataframe(comparison_df.sort_values(by="Persentase Karkas (%)", ascending=False), 
                     hide_index=True,
                     use_container_width=True,
                     column_config={
                         "Persentase Karkas (%)": st.column_config.NumberColumn(format="%.1f%%"),
                         "Persentase Daging dari Berat Hidup (%)": st.column_config.NumberColumn(format="%.1f%%"),
                         "Berat Karkas (kg)": st.column_config.NumberColumn(format="%.2f kg"),
                         "Berat Daging (kg)": st.column_config.NumberColumn(format="%.2f kg")
                     })
    
        # Visualisasi perbandingan
        fig = go.Figure()
    
        # Tambahkan bar untuk persentase karkas
        fig.add_trace(go.Bar(
            x=breed_names,
            y=karkas_percents,
            name='Persentase Karkas',
            marker_color='skyblue',
            text=[f"{p:.1f}%" for p in karkas_percents],
            textposition='auto'
        ))
    
        # Tambahkan bar untuk persentase daging
        fig.add_trace(go.Bar(
            x=breed_names,
            y=meat_percents,
            name='Persentase Daging dari Berat Hidup',
            marker_color='salmon',
            text=[f"{p:.1f}%" for p in meat_percents],
            textposition='auto'
        ))
    
        # Highlight bangsa saat ini
        selected_idx = breed_names.index(bangsa_ternak)
    
        # Update layout
        fig.update_layout(
            title=f"Perbandingan Persentase Karkas dan Daging Antar Bangsa {jenis_ternak}",
            xaxis_title="Bangsa",
            yaxis_title="Persentase (%)",
            barmode='group',
            xaxis={'categoryorder':'total descending'}
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # Tambahkan informasi tambahan
        st.info("""
        ##### Penjelasan Komponen Hasil Pemotongan:
    
        **Karkas** adalah bagian dari tubuh ternak yang telah disembelih setelah dipisahkan dari kepala, kaki, kulit, ekor, 
        organ dalam (jeroan) dan darah. Pada sapi, karkas terdiri dari daging, tulang, dan lemak.
    
        **Non-karkas** adalah semua bagian tubuh selain karkas, terdiri dari:
        - Kepala, kulit, kaki, ekor
        - Organ dalam (jantung, hati, paru-paru, limpa)
        - Saluran pencernaan (lambung, usus)
        - Darah dan lemak non-karkas
    
        **Daging** adalah bagian utama dari karkas yang dapat dikonsumsi, tidak termasuk tulang dan lemak.
    
        > **Catatan**: Nilai-nilai di atas adalah prediksi berdasarkan persentase rata-rata untuk setiap bangsa dan jenis kelamin ternak.
        > Hasil aktual dapat bervariasi tergantung umur, kondisi, tingkat kegemukan, dan faktor lainnya.
        """)
    
        # Visualisasi Data Detail (ekspansi dari fitur sebelumnya)
        st.subheader("Visualisasi Data Detail")
    
        # Tampilkan tabs untuk berbagai visualisasi detail
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Grafik Dimensi & Berat", 
            "Distribusi Berat", 
            "Perbandingan Rumus",
            "Perbandingan Bangsa"
        ])
    
        with viz_tab1:
            # Grafik hubungan dimensi dan berat
            col1, col2 = st.columns(2)
        
            with col1:
                # Grafik hubungan lingkar dada dan berat badan
                ld_range = np.linspace(chest_range['min'] * 0.9, chest_range['max'] * 1.1, 50)
                bb_range = [hitung_berat_badan(ld, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)[0] for ld in ld_range]
            
                fig1, ax1 = plt.subplots()
                ax1.plot(ld_range, bb_range)
                ax1.scatter([lingkar_dada], [berat_badan], color='red', s=100)
            
                # Tambahkan area rentang normal
                ax1.axvspan(chest_range['min'], chest_range['max'], alpha=0.2, color='green', label=f'Rentang normal {bangsa_ternak}')
            
                ax1.set_xlabel('Lingkar Dada (cm)')
                ax1.set_ylabel('Berat Badan (kg)')
                ax1.set_title('Hubungan Lingkar Dada dan Berat Badan')
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)
        
            with col2:
                # Grafik hubungan panjang badan dan berat badan
                pb_range = np.linspace(length_range['min'] * 0.9, length_range['max'] * 1.1, 50)
                bb_range = [hitung_berat_badan(lingkar_dada, pb, jenis_ternak, bangsa_ternak, jenis_kelamin)[0] for pb in pb_range]
            
                fig2, ax2 = plt.subplots()
                ax2.plot(pb_range, bb_range)
                ax2.scatter([panjang_badan], [berat_badan], color='red', s=100)
            
                # Tambahkan area rentang normal
                ax2.axvspan(length_range['min'], length_range['max'], alpha=0.2, color='green', label=f'Rentang normal {bangsa_ternak}')
            
                ax2.set_xlabel('Panjang Badan (cm)')
                ax2.set_ylabel('Berat Badan (kg)')
                ax2.set_title('Hubungan Panjang Badan dan Berat Badan')
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)
        
            # Tabel perbandingan dengan variasi ukuran
            st.subheader("Estimasi Berat dengan Variasi Dimensi Tubuh")
            data = []
        
            # Variasi lingkar dada dan panjang badan (±10%)
            ld_variations = [lingkar_dada * 0.9, lingkar_dada, lingkar_dada * 1.1]
            pb_variations = [panjang_badan * 0.9, panjang_badan, panjang_badan * 1.1]
        
            for ld in ld_variations:
                for pb in pb_variations:
                    bb, _, _ = hitung_berat_badan(ld, pb, jenis_ternak, bangsa_ternak, jenis_kelamin)
                    data.append({
                        "Lingkar Dada (cm)": f"{ld:.1f}",
                        "Panjang Badan (cm)": f"{pb:.1f}",
                        "Berat Badan (kg)": f"{bb:.2f}",
                        "Persentase Perubahan (%)": f"{((bb/berat_badan)-1)*100:.1f}%"
                    })
        
            # Tampilkan tabel dengan highlight
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with viz_tab2:
            # Visualisasi distribusi berat badan
            st.write("##### Distribusi Berat Badan untuk Bangsa dan Jenis Kelamin")
            st.write("Grafik ini menunjukkan distribusi berat umum untuk bangsa dan jenis kelamin ternak ini, dan dimana posisi ternak Anda berada dalam distribusi tersebut.")
        
            # Buat visualisasi distribusi berat
            weight_dist_fig = create_weight_distribution_chart(jenis_ternak, bangsa_ternak, jenis_kelamin, berat_badan)
            st.plotly_chart(weight_dist_fig, use_container_width=True)
        
            # Tambahkan penjelasan tentang distribusi
            breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
        
            # Tentukan kategori berat (ringan, sedang, berat)
            if jenis_ternak == "Sapi":
                if jenis_kelamin == "Jantan":
                    weight_ranges = {"ringan": 300, "sedang": 600, "berat": 900}
                else:
                    weight_ranges = {"ringan": 250, "sedang": 450, "berat": 700}
            elif jenis_ternak == "Kambing":
                if jenis_kelamin == "Jantan":
                    weight_ranges = {"ringan": 30, "sedang": 60, "berat": 90}
                else:
                    weight_ranges = {"ringan": 25, "sedang": 45, "berat": 70}
            else:  # Domba
                if jenis_kelamin == "Jantan":
                    weight_ranges = {"ringan": 35, "sedang": 70, "berat": 120}
                else:
                    weight_ranges = {"ringan": 30, "sedang": 60, "berat": 90}
        
            # Sesuaikan dengan faktor bangsa
            factor = breed_data["factor"]
            for key in weight_ranges:
                weight_ranges[key] = weight_ranges[key] * factor
        
            # Tentukan kategori berat saat ini
            if berat_badan < weight_ranges["ringan"]:
                weight_category = "ringan"
            elif berat_badan < weight_ranges["sedang"]:
                weight_category = "sedang"
            elif berat_badan < weight_ranges["berat"]:
                weight_category = "berat"
            else:
                weight_category = "sangat berat"
        
            st.info(f"""
            ##### Interpretasi Hasil:
        
            Berdasarkan berat badan yang diprediksi ({berat_badan:.2f} kg), ternak Anda termasuk ke dalam **kategori {weight_category}** untuk {bangsa_ternak} {jenis_kelamin}.
        
            **Penjelasan Kategori**:
            - Ringan: < {weight_ranges['ringan']:.0f} kg
            - Sedang: {weight_ranges['ringan']:.0f} - {weight_ranges['sedang']:.0f} kg
            - Berat: {weight_ranges['sedang']:.0f} - {weight_ranges['berat']:.0f} kg
            - Sangat Berat: > {weight_ranges['berat']:.0f} kg
            """)

        with viz_tab3:
            # Perbandingan hasil dari berbagai rumus
            st.write("##### Perbandingan Hasil dari Berbagai Rumus Perhitungan")
            st.write("Berat badan yang sama dapat dihitung dengan berbagai rumus yang berbeda. Berikut perbandingan hasil perhitungan dari berbagai rumus yang tersedia untuk jenis ternak yang dipilih.")
        
            # Dapatkan hasil dari berbagai rumus
            formula_results = compare_formulas(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin, bangsa_ternak)
        
            # Buat dataframe untuk visualisasi
            formula_names = []
            raw_weights = []
            corrected_weights = []
            formula_texts = []
            descriptions = []
        
            for formula_name, result in formula_results.items():
                formula_names.append(formula_name)
                raw_weights.append(result["raw_weight"])
                corrected_weights.append(result["corrected_weight"])
                formula_texts.append(result["formula"])
                descriptions.append(result["description"])
        
            # Buat tabel perbandingan
            formulas_df = pd.DataFrame({
                "Nama Rumus": formula_names,
                "Formula": formula_texts,
                "Berat Dasar (kg)": [f"{w:.2f}" for w in raw_weights],
                "Berat Terkoreksi (kg)": [f"{w:.2f}" for w in corrected_weights],
                "Deskripsi": descriptions
            })
        
            # Tampilkan tabel
            st.dataframe(formulas_df, use_container_width=True, hide_index=True)
        
            # Buat visualisasi perbandingan rumus
            fig = go.Figure()
        
            # Tambahkan batang untuk raw weight
            fig.add_trace(go.Bar(
                x=formula_names, 
                y=raw_weights,
                name='Berat Dasar',
                marker_color='skyblue',
                text=[f"{w:.1f} kg" for w in raw_weights],
                textposition='auto'
            ))
        
            # Tambahkan batang untuk corrected weight
            fig.add_trace(go.Bar(
                x=formula_names, 
                y=corrected_weights,
                name='Berat Terkoreksi',
                marker_color='orangered',
                text=[f"{w:.1f} kg" for w in corrected_weights],
                textposition='auto'
            ))
        
            # Tambahkan garis untuk berat yang dihitung
            fig.add_shape(
                type="line",
                x0=-0.5, 
                y0=berat_badan, 
                x1=len(formula_names)-0.5, 
                y1=berat_badan,
                line=dict(color="green", width=2, dash="dash")
            )
        
            # Tambahkan anotasi untuk berat yang dihitung
            fig.add_annotation(
                x=len(formula_names)-0.5,
                y=berat_badan,
                xshift=10,
                text=f"Berat Saat Ini: {berat_badan:.1f} kg",
                showarrow=False,
                font=dict(color="green", size=12),
                bgcolor="white",
                bordercolor="green",
                borderwidth=1
            )
        
            # Konfigurasi layout
            fig.update_layout(
                title=f"Perbandingan Hasil Perhitungan Berbagai Rumus",
                xaxis_title="Rumus Perhitungan",
                yaxis_title="Berat Badan (kg)",
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1
                ),
                margin=dict(t=80, b=60, l=40, r=40)
            )
        
            # Tampilkan grafik
            st.plotly_chart(fig, use_container_width=True)
        
            # Tambahkan penjelasan
            st.info("""
            ##### Penjelasan Perbandingan Rumus:
        
            **Berat Dasar** adalah hasil perhitungan murni menggunakan rumus tanpa faktor koreksi. 
        
            **Berat Terkoreksi** adalah hasil setelah menerapkan faktor koreksi bangsa dan jenis kelamin.
        
            Perbedaan hasil antar rumus disebabkan oleh:
            1. Perbedaan konstanta perhitungan yang disesuaikan dengan tipe ternak
            2. Perbedaan metode perhitungan yang mempertimbangkan karakteristik fisik ternak yang berbeda
            """)

        with viz_tab4:
            # Perbandingan berat antar bangsa
            st.write("##### Perbandingan Berat Antar Bangsa Ternak")
            st.write("Grafik ini membandingkan berat badan yang dihasilkan pada berbagai bangsa ternak dengan ukuran lingkar dada dan panjang badan yang sama.")
        
            # Buat visualisasi perbandingan bangsa
            breed_comparison_fig = create_breed_comparison_chart(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin)
            st.plotly_chart(breed_comparison_fig, use_container_width=True)
        
            # Tambahkan penjelasan
            st.info("""
            ##### Penjelasan Perbandingan Bangsa:
        
            Grafik di atas menunjukkan bagaimana berat badan bervariasi antar bangsa ternak meskipun dengan ukuran lingkar dada dan panjang badan yang sama. Hal ini disebabkan oleh:
        
            1. **Karakteristik fisik bangsa** - Setiap bangsa memiliki konformasi tubuh, kepadatan otot, dan distribusi lemak yang berbeda
            2. **Rumus yang digunakan** - Bangsa yang berbeda sering menggunakan rumus perhitungan yang berbeda
            3. **Faktor koreksi** - Faktor koreksi spesifik diterapkan untuk setiap bangsa
        
            Perbandingan ini berguna untuk memahami potensi produksi dari berbagai bangsa ternak dan membantu dalam keputusan pemilihan bangsa untuk program peternakan.
            """)

        # Tabel perbandingan
        st.subheader("Tabel Prediksi dengan Variasi Ukuran")
    
        # Fungsi untuk membuat tabel prediksi berat dengan berbagai variasi ukuran
        def create_prediction_table(lingkar_dada, panjang_badan, jenis_ternak, bangsa, jenis_kelamin, steps=5, variation_percent=15):
            """
            Membuat tabel prediksi berat badan dengan variasi ukuran lingkar dada dan panjang badan
        
            Args:
                lingkar_dada (float): Ukuran lingkar dada saat ini (cm)
                panjang_badan (float): Ukuran panjang badan saat ini (cm)
                jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
                bangsa (str): Bangsa ternak
                jenis_kelamin (str): Jenis kelamin ternak
                steps (int): Jumlah langkah variasi (default=5)
                variation_percent (float): Persentase variasi dari nilai tengah (default=15%)
            
            Returns:
                pd.DataFrame: DataFrame berisi tabel prediksi berat dengan variasi ukuran
            """
            # Tentukan rentang variasi
            ld_min = lingkar_dada * (1 - variation_percent/100)
            ld_max = lingkar_dada * (1 + variation_percent/100)
            pb_min = panjang_badan * (1 - variation_percent/100)
            pb_max = panjang_badan * (1 + variation_percent/100)
        
            # Buat array variasi ukuran
            ld_values = np.linspace(ld_min, ld_max, steps)
            pb_values = np.linspace(pb_min, pb_max, steps)
        
            # Format untuk nama kolom (lingkar dada)
            ld_headers = [f"LD: {ld:.1f} cm" for ld in ld_values]
        
            # Buat dataframe untuk menyimpan hasil
            results = []
        
            # Hitung berat untuk setiap kombinasi
            for pb in pb_values:
                row = {"Panjang Badan (cm)": f"{pb:.1f}"}
            
                for i, ld in enumerate(ld_values):
                    bb, _, _ = hitung_berat_badan(ld, pb, jenis_ternak, bangsa, jenis_kelamin)
                    row[ld_headers[i]] = f"{bb:.1f} kg"
            
                results.append(row)
        
            # Kembalikan DataFrame
            return pd.DataFrame(results)
    
        # Tampilkan tabel prediksi berat dengan berbagai variasi ukuran
        st.write("""
        Tabel di bawah ini menunjukkan prediksi berat badan ternak dengan berbagai variasi ukuran lingkar dada (LD) 
        dan panjang badan (PB). Gunakan tabel ini untuk memperkirakan berat ternak dengan rentang ukuran yang lebih luas
        atau untuk memahami bagaimana perubahan kecil pada pengukuran dapat mempengaruhi hasil prediksi berat.
        """)
    
        # Buat container untuk memperbarui konten tabel saat slider berubah
        table_container = st.container()
    
        # Opsi untuk kustomisasi tabel
        col1, col2 = st.columns([1, 1])
        with col1:
            variation_percent = st.slider("Rentang Variasi (%)", min_value=5, max_value=30, value=15, 
                                          help="Persentase variasi ukuran dari nilai tengah", key="variation_percent_slider")
        with col2:
            steps = st.slider("Jumlah Langkah Variasi", min_value=3, max_value=9, value=5, step=2,
                              help="Jumlah langkah variasi ukuran (kolom dan baris)", key="steps_slider")
    
        # Buat dan tampilkan tabel prediksi dalam container yang akan diperbarui saat slider berubah
        with table_container:
            # Buat tabel baru setiap kali slider berubah
            prediction_table = create_prediction_table(
                lingkar_dada=lingkar_dada,
                panjang_badan=panjang_badan,
                jenis_ternak=jenis_ternak,
                bangsa=bangsa_ternak,
                jenis_kelamin=jenis_kelamin,
                steps=steps,
                variation_percent=variation_percent
            )
        
            # Tampilkan tabel dengan highlight pada nilai tengah
            st.dataframe(prediction_table, use_container_width=True, hide_index=True)
    
        # Tambahkan penjelasan dan tips penggunaan
        st.info("""
        ##### Cara Menggunakan Tabel Prediksi:
    
        1. **Bandingkan rentang** - Lihat bagaimana berat badan berubah dengan variasi ukuran lingkar dada dan panjang badan
        2. **Antisipasi pertumbuhan** - Gunakan untuk memperkirakan pertambahan berat jika ukuran tubuh ternak bertambah
        3. **Koreksi pengukuran** - Jika tidak yakin dengan pengukuran awal, lihat rentang beratnya pada variasi ukuran
        4. **Nilai optimal** - Identifikasi target ukuran tubuh untuk mencapai berat badan yang diinginkan
    
        > **Tips**: Pengukuran lingkar dada memiliki pengaruh lebih besar terhadap berat badan dibandingkan dengan panjang badan,
        > karena dalam rumus perhitungan, lingkar dada dikuadratkan sedangkan panjang badan tidak.
        """)
    
        # Buat container untuk memperbarui heatmap saat slider berubah
        heatmap_container = st.container()
    
        # Tampilkan visualisasi heatmap berat badan
        with heatmap_container:
            st.subheader("Peta Panas Prediksi Berat Badan")
            st.write("Visualisasi di bawah ini menunjukkan hubungan antara lingkar dada, panjang badan, dan prediksi berat badan dalam bentuk peta panas (heatmap).")
        
            # Buat array untuk heatmap (gunakan nilai slider terbaru)
            ld_values = np.linspace(lingkar_dada * (1 - variation_percent/100), 
                                   lingkar_dada * (1 + variation_percent/100), 
                                   20)  # Lebih banyak titik untuk visualisasi yang lebih halus
            pb_values = np.linspace(panjang_badan * (1 - variation_percent/100), 
                                   panjang_badan * (1 + variation_percent/100), 
                                   20)
        
            # Buat grid untuk heatmap
            ld_grid, pb_grid = np.meshgrid(ld_values, pb_values)
            weights = np.zeros(ld_grid.shape)
        
            # Hitung berat untuk setiap kombinasi ukuran
            for i in range(ld_grid.shape[0]):
                for j in range(ld_grid.shape[1]):
                    weights[i, j], _, _ = hitung_berat_badan(ld_grid[i, j], pb_grid[i, j], 
                                                            jenis_ternak, bangsa_ternak, jenis_kelamin)
        
            # Buat heatmap dengan Plotly
            fig = go.Figure(data=go.Heatmap(
                z=weights,
                x=ld_values,
                y=pb_values,
                colorscale='Viridis',
                colorbar=dict(title='Berat (kg)')
            ))
        
            # Tambahkan marker untuk nilai saat ini
            fig.add_trace(go.Scatter(
                x=[lingkar_dada],
                y=[panjang_badan],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Ukuran Saat Ini'
            ))
        
            # Konfigurasi layout
            fig.update_layout(
                title=f"Peta Panas Prediksi Berat {jenis_ternak} {bangsa_ternak} ({jenis_kelamin})<br>Rentang Variasi: {variation_percent}%, Langkah: {steps}",
                xaxis_title="Lingkar Dada (cm)",
                yaxis_title="Panjang Badan (cm)",
                height=500
            )
        
            # Tampilkan heatmap
            st.plotly_chart(fig, use_container_width=True)




st.markdown("---")
st.subheader("8. Arsip, Riwayat, dan Mode Banyak Ternak")
riwayat_tab, batch_tab = st.tabs(["📋 Riwayat & Unduhan", "📤 Mode Banyak Ternak"])

with riwayat_tab:
    if st.session_state.calculation_history:
        history_df = pd.DataFrame(st.session_state.calculation_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Download Riwayat CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="riwayat_prediksi_ternak.csv",
            mime="text/csv"
        )
        if st.button("Hapus Riwayat"):
            st.session_state.calculation_history = []
            st.success("Riwayat perhitungan berhasil dihapus.")
    else:
        st.info("Belum ada riwayat. Klik tombol Hitung Berat Badan untuk menyimpan hasil ke riwayat.")

with batch_tab:
    st.markdown("""
    Gunakan fitur ini untuk menghitung banyak ternak sekaligus dari file CSV atau Excel.

    **Alur mode banyak ternak:**
    1. Download template CSV.
    2. Isi data ternak sesuai kolom.
    3. Upload kembali file CSV/Excel.
    4. Download hasil prediksi.

    Kolom wajib:
    - `Jenis Ternak`
    - `Bangsa Ternak`
    - `Jenis Kelamin`
    - `Lingkar Dada`
    - `Panjang Badan`

    Kolom opsional:
    - `BCS / Kondisi Tubuh` (`Tidak dinilai`, `1 - Sangat Kurus`, `2 - Kurus`, `3 - Sedang/Ideal`, `4 - Gemuk`, `5 - Sangat Gemuk`)
    - `Kelas Pasar` (`Otomatis`, `Kelas A / Super`, `Kelas B / Normal`, `Kelas C / Kurus`)
    - `Margin Error (%)`
    - `Harga per Kg`
    - `Harga per Kg Karkas`
    - `Harga per Kg Daging`
    - `Harga Beli / Modal`
    - `Biaya Pakan per Hari`
    - `Lama Pemeliharaan (Hari)`
    - `Biaya Obat/Vitamin`
    - `Biaya Transportasi`
    - `Biaya Lain-lain`

    Jika kolom harga dikosongkan atau diisi 0, aplikasi memakai harga default berdasarkan jenis, bangsa, dan kelas pasar ternak.
    """)

    template_rows = [
        {
            "Jenis Ternak": "Sapi",
            "Bangsa Ternak": "Sapi Bali",
            "Jenis Kelamin": "Jantan",
            "Lingkar Dada": 175,
            "Panjang Badan": 150,
        },
        {
            "Jenis Ternak": "Sapi",
            "Bangsa Ternak": "Sapi Limousin",
            "Jenis Kelamin": "Jantan",
            "Lingkar Dada": 215,
            "Panjang Badan": 190,
        },
        {
            "Jenis Ternak": "Kambing",
            "Bangsa Ternak": "Kambing Boer",
            "Jenis Kelamin": "Betina",
            "Lingkar Dada": 85,
            "Panjang Badan": 75,
        },
        {
            "Jenis Ternak": "Domba",
            "Bangsa Ternak": "Domba Garut",
            "Jenis Kelamin": "Jantan",
            "Lingkar Dada": 80,
            "Panjang Badan": 70,
        },
    ]

    for row in template_rows:
        status_template, _ = get_size_status(
            row["Lingkar Dada"],
            row["Panjang Badan"],
            row["Jenis Ternak"],
            row["Bangsa Ternak"],
        )
        kelas_template, _, multiplier_template = get_market_class(status_template, "Otomatis")
        prices = get_latest_price_defaults(row["Jenis Ternak"], row["Bangsa Ternak"])
        prices = apply_market_class_to_prices(prices, multiplier_template, kelas_template)

        row["BCS / Kondisi Tubuh"] = "Tidak dinilai"
        row["Kelas Pasar"] = "Otomatis"
        row["Margin Error (%)"] = 10
        row["Harga per Kg"] = prices["harga_bobot_hidup"]
        row["Harga per Kg Karkas"] = prices["harga_karkas"]
        row["Harga per Kg Daging"] = prices["harga_daging"]
        row["Harga Beli / Modal"] = 0
        row["Biaya Pakan per Hari"] = 0
        row["Lama Pemeliharaan (Hari)"] = 0
        row["Biaya Obat/Vitamin"] = 0
        row["Biaya Transportasi"] = 0
        row["Biaya Lain-lain"] = 0

    template_df = pd.DataFrame(template_rows)

    st.download_button(
        label="⬇️ Download Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_data_ternak.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)

            batch_result_df = process_batch_dataframe(input_df)
            st.success("Data berhasil diproses.")
            st.dataframe(batch_result_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="⬇️ Download Hasil CSV",
                data=batch_result_df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_banyak_ternak.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Gagal memproses file: {exc}")

# Footer with adaptive light/dark styling
st.markdown(f"""
<div class="footer-card">
    <p>
        &copy; {current_year} Developed by:
        <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank">
            Galuh Adi Insani
        </a>
        with <span style="color:#e25555">❤️</span>
    </p>
    <p class="muted">All rights reserved.</p>
</div>
""", unsafe_allow_html=True)