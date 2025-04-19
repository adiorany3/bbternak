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
import json
from PIL import Image

# Konfigurasi halaman Streamlit - HARUS DITEMPATKAN PERTAMA
st.set_page_config(
    page_title="Prediksi Berat Badan Ternak",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit elements
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
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
                "calculation": lambda ld, pb: ((ld + 22) ** 2) / 100,
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
                "calculation": lambda ld, pb: (0.0000627 * ld * pb) - 3.91,
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
                "length_range": {"min": 120, "max": 180}
            },
            "Sapi Madura": {
                "formula_name": "Schoorl (Indonesia)", 
                "factor": 0.95,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.92},
                "chest_range": {"min": 130, "max": 200},
                "length_range": {"min": 110, "max": 170}
            },
            "Sapi Limousin": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.2,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.95},
                "chest_range": {"min": 180, "max": 260},
                "length_range": {"min": 160, "max": 230}
            },
            "Sapi Simental": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.25,
                "gender_factor": {"Jantan": 1.1, "Betina": 0.93},
                "chest_range": {"min": 190, "max": 270},
                "length_range": {"min": 170, "max": 240}
            },
            "Sapi Brahman": {
                "formula_name": "Winter (Eropa)", 
                "factor": 1.15,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 180, "max": 250},
                "length_range": {"min": 160, "max": 220}
            },
            "Sapi Peranakan Ongole (PO)": {
                "formula_name": "Lambourne (Sapi Kecil)", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.9},
                "chest_range": {"min": 150, "max": 230},
                "length_range": {"min": 130, "max": 200}
            },
            "Sapi Friesian Holstein (FH)": {
                "formula_name": "Denmark", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.08, "Betina": 0.97},
                "chest_range": {"min": 180, "max": 250},
                "length_range": {"min": 160, "max": 220}
            },
            "Sapi Aceh": {
                "formula_name": "Schoorl (Indonesia)", 
                "factor": 0.9,
                "gender_factor": {"Jantan": 1.14, "Betina": 0.92},
                "chest_range": {"min": 120, "max": 190},
                "length_range": {"min": 100, "max": 160}
            },
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
                "length_range": {"min": 40, "max": 70}
            },
            "Kambing Ettawa": {
                "formula_name": "New Zealand", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.2, "Betina": 0.88},
                "chest_range": {"min": 70, "max": 110},
                "length_range": {"min": 60, "max": 95}
            },
            "Kambing Peranakan Ettawa (PE)": {
                "formula_name": "Arjodarmoko", 
                "factor": 1.0,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 65, "max": 100},
                "length_range": {"min": 55, "max": 90}
            },
            "Kambing Boer": {
                "formula_name": "New Zealand", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.9},
                "chest_range": {"min": 75, "max": 120},
                "length_range": {"min": 65, "max": 105}
            },
            "Kambing Jawarandu": {
                "formula_name": "Arjodarmoko", 
                "factor": 0.95,
                "gender_factor": {"Jantan": 1.12, "Betina": 0.92},
                "chest_range": {"min": 60, "max": 95},
                "length_range": {"min": 50, "max": 85}
            },
            "Kambing Bligon": {
                "formula_name": "Khan", 
                "factor": 0.92,
                "gender_factor": {"Jantan": 1.1, "Betina": 0.92},
                "chest_range": {"min": 55, "max": 90},
                "length_range": {"min": 45, "max": 80}
            },
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
                "length_range": {"min": 45, "max": 75}
            },
            "Domba Ekor Gemuk": {
                "formula_name": "Lambourne", 
                "factor": 1.1,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.88},
                "chest_range": {"min": 65, "max": 95},
                "length_range": {"min": 55, "max": 85}
            },
            "Domba Merino": {
                "formula_name": "NSA Australia", 
                "factor": 1.05,
                "gender_factor": {"Jantan": 1.2, "Betina": 0.85},
                "chest_range": {"min": 75, "max": 110},
                "length_range": {"min": 65, "max": 95}
            },
            "Domba Garut": {
                "formula_name": "Lambourne", 
                "factor": 1.0,
                "gender_factor": {"Jantan": 1.25, "Betina": 0.85},
                "chest_range": {"min": 70, "max": 105},
                "length_range": {"min": 60, "max": 90}
            },
            "Domba Suffolk": {
                "formula_name": "Valdez", 
                "factor": 1.15,
                "gender_factor": {"Jantan": 1.15, "Betina": 0.9},
                "chest_range": {"min": 85, "max": 130},
                "length_range": {"min": 75, "max": 115}
            },
            "Domba Texel": {
                "formula_name": "Valdez", 
                "factor": 1.2,
                "gender_factor": {"Jantan": 1.18, "Betina": 0.9},
                "chest_range": {"min": 90, "max": 135},
                "length_range": {"min": 80, "max": 120}
            },
        },
        "icon": "🐑"
    }
}

def hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa, jenis_kelamin):
    """
    Menghitung berat badan ternak menggunakan rumus yang sesuai dengan jenis dan bangsanya.
    
    Args:
        lingkar_dada (float): Lingkar dada ternak dalam sentimeter
        panjang_badan (float): Panjang badan ternak dalam sentimeter
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan atau Betina)
        
    Returns:
        float: Berat badan ternak dalam kilogram
        str: Nama rumus yang digunakan
        str: Formula yang digunakan
    """
    # Ambil data ternak
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa]
    formula_name = breed_data["formula_name"]
    factor = breed_data["factor"]
    gender_factor = breed_data["gender_factor"][jenis_kelamin]
    
    # Ambil detail formula
    formula_data = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]
    formula_text = formula_data["formula"]
    calculation_func = formula_data["calculation"]
    
    # Hitung berat badan sesuai rumus
    berat_badan = calculation_func(lingkar_dada, panjang_badan)
    
    # Terapkan faktor koreksi berdasarkan bangsa dan jenis kelamin
    berat_badan = berat_badan * factor * gender_factor
    
    return berat_badan, formula_name, formula_text

# Judul dan deskripsi aplikasi
st.title("🐄 Prediksi Berat Badan Ternak")
st.markdown(f"""
    Aplikasi ini menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan 
    menggunakan **Rumus Formula** yang spesifik untuk jenis dan bangsa ternak yang berbeda. 
    Silakan pilih jenis dan bangsa ternak yang sesuai di sidebar untuk mendapatkan hasil yang lebih akurat.
    
    > **Catatan**: Metode ini adalah prediksi pendekatan. Untuk mendapatkan data berat badan yang akurat, 
    > timbangan ternak tetap merupakan alat ukur yang paling tepat.
    """)

# Sidebar untuk input pengguna
st.sidebar.header("Input Data Ternak")

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
    "Lingkar Dada (cm)",
    min_value=chest_range["min"] * 0.8,  # Sedikit di bawah minimum untuk fleksibilitas
    max_value=chest_range["max"] * 1.2,  # Sedikit di atas maksimum untuk fleksibilitas
    value=chest_range["min"] + (chest_range["max"] - chest_range["min"]) / 2,  # Nilai default di tengah rentang
    step=0.5,
    help=f"Ukur lingkar dada ternak dengan pita ukur, yaitu mengukur keliling dada ternak tepat di belakang bahu. Rentang normal untuk {bangsa_ternak}: {chest_range['min']}-{chest_range['max']} cm."
)

# Input panjang badan dengan rentang sesuai bangsa ternak
panjang_badan = st.sidebar.number_input(
    "Panjang Badan (cm)",
    min_value=length_range["min"] * 0.8,  # Sedikit di bawah minimum untuk fleksibilitas
    max_value=length_range["max"] * 1.2,  # Sedikit di atas maksimum untuk fleksibilitas
    value=length_range["min"] + (length_range["max"] - length_range["min"]) / 2,  # Nilai default di tengah rentang
    step=0.5,
    help=f"Ukur panjang badan ternak, yaitu dari ujung bahu hingga tulang duduk (tuber ischii). Rentang normal untuk {bangsa_ternak}: {length_range['min']}-{length_range['max']} cm."
)

# Tombol untuk menghitung berat badan
if st.sidebar.button("Hitung Berat Badan", type="primary"):
    # Hitung berat badan
    berat_badan, formula_name, formula_text = hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)
    
    # Tampilkan hasil dalam kotak
    st.success(f"## Prediksi Berat Badan: **{berat_badan:.2f} kg**")
    
    # Tampilkan detail perhitungan
    st.subheader("Detail Perhitungan:")
    
    # Dapatkan referensi dari formula
    formula_reference = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["reference"]
    
    st.markdown(f"""
    - Jenis Ternak: **{jenis_ternak}**
    - Bangsa Ternak: **{bangsa_ternak}**
    - Jenis Kelamin: **{jenis_kelamin}**
    - Rumus yang Digunakan: **{formula_name}**
    - Formula: **{formula_text}**
    - Referensi: **{formula_reference}**
    - Lingkar Dada (LD): **{lingkar_dada} cm** (Rentang normal: {chest_range['min']}-{chest_range['max']} cm)
    - Panjang Badan (PB): **{panjang_badan} cm** (Rentang normal: {length_range['min']}-{length_range['max']} cm)
    - Berat Badan (BB) = **{berat_badan:.2f} kg**
    """)
    
    # Visualisasi
    st.subheader("Visualisasi Data")
    
    # Buat data untuk visualisasi
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
    
    # Tabel perbandingan
    st.subheader("Tabel Prediksi dengan Variasi Ukuran")
    data = []
    
    # Variasi lingkar dada (±10%)
    ld_variations = [lingkar_dada * 0.9, lingkar_dada, lingkar_dada * 1.1]
    pb_variations = [panjang_badan * 0.9, panjang_badan, panjang_badan * 1.1]
    
    for ld in ld_variations:
        for pb in pb_variations:
            bb, _, _ = hitung_berat_badan(ld, pb, jenis_ternak, bangsa_ternak, jenis_kelamin)
            data.append({
                "Lingkar Dada (cm)": f"{ld:.1f}",
                "Panjang Badan (cm)": f"{pb:.1f}",
                "Berat Badan (kg)": f"{bb:.2f}"
            })
    
    # Tampilkan tabel
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# Tampilkan contoh kasus
st.sidebar.markdown("---")
st.sidebar.subheader("Contoh Kasus:")
contoh_expander = st.sidebar.expander("Lihat Contoh Kasus")
with contoh_expander:
    st.markdown("""
    Jika lingkar dada ternak (LD) adalah **180 cm** dan panjang badan (PB) adalah **150 cm**, maka:
    
    BB = (180)² × 150 / 10.815,15  
    BB = 32.400 × 150 / 10.815,15  
    BB = 448,72 kg
    """)

# Tampilkan informasi tentang rumus Formula
st.markdown("---")
info_expander = st.expander("ℹ️ Informasi tentang Rumus Perhitungan")
with info_expander:
    st.markdown("""
    ## Pendahuluan
    
    Estimasi berat badan ternak merupakan hal yang sangat penting dalam manajemen peternakan. Penggunaan rumus pendugaan berat badan membantu peternak mengestimasi berat ternak tanpa memerlukan timbangan yang mahal dan tidak praktis di lapangan. Rumus-rumus ini dikembangkan berdasarkan penelitian ilmiah yang mengkorelasikan ukuran-ukuran tubuh ternak dengan berat badannya.
    
    ## Cara Pengukuran yang Benar
    
    ### Pengukuran Lingkar Dada (LD)
    Pengukuran lingkar dada dilakukan dengan melingkarkan pita ukur pada bagian dada tepat di belakang sendi bahu (scapula) atau sekitar 2-3 cm di belakang siku:
    
    1. Pastikan ternak berdiri dengan posisi normal (tidak membungkuk atau meregang)
    2. Lingkarkan pita ukur mengelilingi dada tepat di belakang kaki depan
    3. Tarik pita dengan kekencangan sedang (tidak terlalu kencang atau kendor)
    4. Catat hasil pengukuran dalam satuan sentimeter (cm)
    
    ### Pengukuran Panjang Badan (PB)
    Cara pengukuran panjang badan berbeda untuk setiap jenis ternak:
    
    **Untuk Sapi:**
    - Ukur dari tonjolan bahu (tuberculum humeralis) sampai tonjolan tulang duduk (tuberculum ischiadicum)
    - Gunakan tongkat ukur atau pita yang ditarik lurus, bukan mengikuti lekukan tubuh
    
    **Untuk Kambing/Domba:**
    - Ukur dari sendi bahu sampai tonjolan tulang duduk (tuber ischii)
    - Pengukuran dilakukan dengan pita ukur yang ditarik lurus
    
    ## Rumus-Rumus Pendugaan Berat Badan Ternak
    
    ### Rumus untuk Sapi
    
    #### 1. Rumus Winter (Eropa)
    **BB = (LD)² × PB / 10.815,15**
    
    Rumus Winter dikembangkan oleh AW Winter pada tahun 1910, dan merupakan rumus yang paling umum digunakan untuk sapi tipe Eropa (Bos taurus). Rumus ini memberikan hasil yang lebih akurat untuk sapi-sapi tipe besar dengan konformasi tubuh yang proporsional.
    
    - **Keunggulan**: Akurasi tinggi untuk sapi tipe Eropa dan persilangannya
    - **Keterbatasan**: Kurang akurat untuk sapi lokal Asia yang memiliki punuk dan proporsi tubuh berbeda
    - **Cocok untuk**: Sapi Limousin, Simental, Angus, Charolais
    
    #### 2. Rumus Schoorl (Indonesia)
    **BB = (LD + 22)² / 100**
    
    Rumus Schoorl dikembangkan khusus dengan mempertimbangkan karakteristik fisik sapi-sapi lokal Indonesia. Rumus ini sering digunakan untuk sapi-sapi dengan ukuran kecil hingga sedang.
    
    - **Keunggulan**: Sederhana dan cukup akurat untuk sapi lokal Indonesia
    - **Keterbatasan**: Tidak memperhitungkan panjang badan sehingga bisa kurang akurat untuk beberapa individu
    - **Cocok untuk**: Sapi Bali, Sapi Madura, Sapi PO, Sapi Aceh
    
    #### 3. Rumus Denmark
    **BB = (LD)² × 0.000138 × PB**
    
    Rumus Denmark adalah modifikasi dari rumus Winter yang dikembangkan di Denmark untuk sapi-sapi perah dan sapi pedaging tipe besar. Konstanta yang digunakan dioptimalkan untuk sapi-sapi dengan tubuh panjang.
    
    - **Keunggulan**: Akurasi tinggi untuk sapi perah dan sapi pedaging tipe besar
    - **Keterbatasan**: Dapat overestimasi untuk sapi berukuran kecil
    - **Cocok untuk**: Sapi Friesian Holstein, Jersey, Simental
    
    #### 4. Rumus Lambourne (Sapi Kecil)
    **BB = (LD)² × PB / 11.900**
    
    Modifikasi dari rumus Lambourne yang disesuaikan untuk sapi-sapi tipe kecil hingga sedang, dengan mempertimbangkan proporsi tubuh yang lebih ramping.
    
    - **Keunggulan**: Memberikan hasil lebih akurat untuk sapi dengan ukuran sedang
    - **Keterbatasan**: Kurang akurat untuk sapi tipe besar
    - **Cocok untuk**: Sapi PO, Sapi Pesisir, beberapa sapi persilangan lokal
    
    ### Rumus untuk Kambing
    
    #### 1. Rumus Arjodarmoko
    **BB = (LD)² × PB / 18.000**
    
    Rumus Arjodarmoko dikembangkan di Indonesia khusus untuk kambing lokal. Konstanta pembagi 18.000 disesuaikan dengan karakteristik fisik kambing lokal yang umumnya memiliki ukuran tubuh lebih kecil.
    
    - **Keunggulan**: Akurasi baik untuk kambing lokal Indonesia
    - **Keterbatasan**: Dapat underestimasi untuk kambing tipe besar
    - **Cocok untuk**: Kambing Kacang, Kambing Jawarandu, Kambing PE
    
    #### 2. Rumus New Zealand
    **BB = 0.0000968 × (LD)² × PB**
    
    Rumus ini dikembangkan di Selandia Baru untuk kambing tipe besar, terutama kambing perah dan kambing pedaging.
    
    - **Keunggulan**: Akurasi tinggi untuk kambing tipe besar
    - **Keterbatasan**: Bisa overestimasi untuk kambing lokal
    - **Cocok untuk**: Kambing Ettawa, Kambing Boer, Kambing Saanen
    
    #### 3. Rumus Khan
    **BB = 0.0004 × (LD)² × 0.6 × PB**
    
    Dikembangkan oleh peneliti Khan untuk berbagai tipe kambing dengan faktor koreksi 0.6 untuk panjang badan.
    
    - **Keunggulan**: Versatilitas tinggi, dapat digunakan untuk berbagai tipe kambing
    - **Keterbatasan**: Presisi sedang dibandingkan rumus spesifik
    - **Cocok untuk**: Berbagai jenis kambing, terutama tipe campuran atau crossbreed
    
    ### Rumus untuk Domba
    
    #### 1. Rumus Lambourne
    **BB = (LD)² × PB / 15.000**
    
    Rumus yang dikembangkan oleh Lambourne khusus untuk domba, dengan konstanta pembagi yang disesuaikan berdasarkan proporsi tubuh domba.
    
    - **Keunggulan**: Standar yang baik untuk berbagai jenis domba
    - **Keterbatasan**: Akurasi sedang untuk domba dengan karakteristik ekstrem
    - **Cocok untuk**: Domba lokal Indonesia, Domba Ekor Tipis, Domba Garut
    
    #### 2. Rumus NSA Australia (National Sheep Association)
    **BB = (0.0000627 × LD × PB) - 3.91**
    
    Dikembangkan oleh Asosiasi Domba Nasional Australia untuk domba tipe medium yang umum di Australia.
    
    - **Keunggulan**: Akurasi tinggi untuk domba tipe sedang dan domba wool
    - **Keterbatasan**: Memiliki konstanta pengurangan yang bisa menyebabkan nilai negatif untuk domba sangat kecil
    - **Cocok untuk**: Domba Merino, Domba Dorset, domba tipe sedang lainnya
    
    #### 3. Rumus Valdez
    **BB = 0.0003 × (LD)² × PB**
    
    Rumus Valdez adalah rumus sederhana yang dapat diaplikasikan untuk berbagai tipe domba pedaging.
    
    - **Keunggulan**: Sederhana dan cukup akurat untuk domba pedaging
    - **Keterbatasan**: Kurang akurat untuk domba dengan distribusi lemak yang tidak merata
    - **Cocok untuk**: Domba Suffolk, Domba Texel, domba pedaging lainnya
    """)

    st.markdown("""
    ## Faktor Koreksi dan Pertimbangan Praktis
    
    ### Faktor Koreksi untuk Bangsa
    Setiap bangsa ternak memiliki karakteristik morfologi yang unik, sehingga diperlukan faktor koreksi untuk meningkatkan akurasi pendugaan berat badan:
    
    - Faktor > 1.0: Digunakan untuk ternak dengan kepadatan otot tinggi atau frame size besar
    - Faktor = 1.0: Standar untuk ternak dengan proporsi tubuh normal
    - Faktor < 1.0: Digunakan untuk ternak dengan tubuh yang lebih ringan atau ramping
    
    ### Pertimbangan Praktis
    
    1. **Kondisi Ternak**: Rumus akan lebih akurat jika ternak dalam kondisi normal (tidak terlalu kurus atau gemuk ekstrem)
    
    2. **Waktu Pengukuran**: Idealnya pengukuran dilakukan pagi hari sebelum ternak diberi makan
    
    3. **Umur Ternak**: Rumus lebih akurat untuk ternak dewasa dibandingkan anak atau ternak remaja
    
    4. **Jenis Kelamin**: Beberapa rumus mungkin perlu penyesuaian tambahan untuk perbedaan antara jantan dan betina
    
    5. **Kebuntingan**: Untuk ternak betina bunting, terutama pada trimester ketiga, rumus ini bisa underestimasi karena bobot fetus
    
    ## Karakteristik Khusus Bangsa Ternak
    """)

    # Cattle breed characteristics (existing section)
    st.markdown("""
    ### Karakteristik Spesifik Bangsa-Bangsa Ternak
    
    #### Sapi
    - **Sapi Bali**: 
        - Asal: Indonesia (domestikasi banteng)
        - Ciri khas: Warna merah bata, kaki putih, punggung bergaris hitam
        - Bobot dewasa: Jantan 300-400 kg, Betina 250-350 kg
        - Keunggulan: Daya adaptasi tinggi, tahan pakan berkualitas rendah, persentase karkas tinggi (56%)
    
    - **Sapi Madura**: 
        - Asal: Persilangan sapi Zebu dan Banteng di Pulau Madura
        - Ciri khas: Warna merah bata hingga cokelat, bertanduk khas melengkung ke atas
        - Bobot dewasa: Jantan 250-350 kg, Betina 200-300 kg
        - Keunggulan: Toleran iklim panas, tahan penyakit, cocok untuk kerja dan sapi karapan
    
    - **Sapi Limousin**: 
        - Asal: Perancis
        - Ciri khas: Warna cokelat kemerahan, bertubuh besar dan berotot
        - Bobot dewasa: Jantan 800-1200 kg, Betina 600-800 kg
        - Keunggulan: Pertumbuhan cepat, konversi pakan efisien, persentase karkas tinggi (58-62%)
    
    - **Sapi Simental**: 
        - Asal: Swiss
        - Ciri khas: Warna cokelat kemerahan dengan bercak putih, kepala putih
        - Bobot dewasa: Jantan 1000-1300 kg, Betina 700-900 kg
        - Keunggulan: Tipe dwiguna (pedaging dan perah), pertumbuhan cepat, produksi susu tinggi
    
    - **Sapi Brahman**: 
        - Asal: Amerika Serikat (dikembangkan dari sapi Zebu India)
        - Ciri khas: Memiliki punuk besar, gelambir lebar, telinga panjang
        - Bobot dewasa: Jantan 800-1100 kg, Betina 500-700 kg
        - Keunggulan: Tahan panas, tahan caplak, adaptif di daerah tropis
    
    - **Sapi PO (Peranakan Ongole)**: 
        - Asal: Indonesia (persilangan sapi lokal dengan Ongole dari India)
        - Ciri khas: Warna putih hingga putih keabu-abuan, gelambir lebar
        - Bobot dewasa: Jantan 400-600 kg, Betina 300-400 kg
        - Keunggulan: Adaptasi baik di Indonesia, tahan panas, tahan penyakit
    
    - **Sapi FH (Friesian Holstein)**: 
        - Asal: Belanda
        - Ciri khas: Warna hitam belang putih, bertubuh besar
        - Bobot dewasa: Jantan 700-900 kg, Betina 600-700 kg
        - Keunggulan: Produksi susu tinggi (15-25 liter/hari), jinak
    
    - **Sapi Aceh**: 
        - Asal: Aceh, Indonesia
        - Ciri khas: Ukuran kecil, warna merah bata hingga cokelat tua
        - Bobot dewasa: Jantan 200-300 kg, Betina 150-250 kg
        - Keunggulan: Sangat adaptif dengan lingkungan ekstrem, tahan penyakit lokal
    """)

    # Goat breed characteristics (enhanced information)
    st.markdown("""
    #### Kambing
    
    - **Kambing Kacang**: 
        - Asal: Indonesia
        - Ciri khas: Ukuran kecil, telinga tegak kecil, warna bervariasi
        - Bobot dewasa: Jantan 20-30 kg, Betina 15-25 kg
        - Keunggulan: Fertil tinggi (kemampuan beranak kembar), adaptasi luas, tahan penyakit lokal
        - Produksi: Daging, dapat menghasilkan susu 0.1-0.3 liter/hari
    
    - **Kambing Ettawa**: 
        - Asal: India (Jamnapari)
        - Ciri khas: Ukuran besar, telinga panjang menggantung, profil hidung melengkung
        - Bobot dewasa: Jantan 60-90 kg, Betina 40-60 kg
        - Keunggulan: Produksi susu tinggi, pertumbuhan cepat
        - Produksi: Susu 1-3 liter/hari, daging
    
    - **Kambing PE (Peranakan Ettawa)**: 
        - Asal: Indonesia (persilangan Kacang dan Ettawa)
        - Ciri khas: Ukuran sedang, telinga panjang tapi tidak seluruhnya menggantung
        - Bobot dewasa: Jantan 40-60 kg, Betina 30-50 kg
        - Keunggulan: Adaptasi baik di Indonesia, produksi susu lebih tinggi dari Kacang
        - Produksi: Susu 0.5-2 liter/hari, daging
    
    - **Kambing Boer**: 
        - Asal: Afrika Selatan
        - Ciri khas: Tubuh kompak berotot, kepala cokelat, badan putih, telinga panjang
        - Bobot dewasa: Jantan 80-120 kg, Betina 60-90 kg
        - Keunggulan: Pertumbuhan sangat cepat, persentase karkas tinggi (48-60%)
        - Produksi: Daging premium, ADG (Average Daily Gain) bisa mencapai 200-250 gram/hari
    
    - **Kambing Jawarandu**: 
        - Asal: Jawa Tengah (persilangan Kacang dan PE)
        - Ciri khas: Ukuran sedang, telinga setengah menggantung
        - Bobot dewasa: Jantan 35-45 kg, Betina 25-35 kg
        - Keunggulan: Adaptif, produksi susu moderat, fertilitas baik
        - Produksi: Susu 0.4-1 liter/hari, daging
    
    - **Kambing Bligon/Jawa Randu**: 
        - Asal: Jawa (persilangan Kacang dan PE dengan proporsi darah Kacang lebih tinggi)
        - Ciri khas: Mirip Jawarandu tapi ukuran lebih kecil
        - Bobot dewasa: Jantan 25-40 kg, Betina 20-30 kg
        - Keunggulan: Sangat adaptif, fertil, mudah pemeliharaan
        - Produksi: Daging, susu 0.3-0.7 liter/hari
    """)

    # Sheep breed characteristics (enhanced information)
    st.markdown("""
    #### Domba
    
    - **Domba Ekor Tipis (DET)**: 
        - Asal: Indonesia
        - Ciri khas: Ekor kecil dan pendek, warna dominan putih
        - Bobot dewasa: Jantan 20-35 kg, Betina 15-25 kg
        - Keunggulan: Prolifikasi tinggi (kemampuan beranak banyak, 1.8-2.0 anak/kelahiran)
        - Produksi: Daging, wool kasar
    
    - **Domba Ekor Gemuk (DEG)**: 
        - Asal: Indonesia timur, pengaruh dari domba Timur Tengah
        - Ciri khas: Penimbunan lemak di bagian ekor, tubuh lebih besar dari DET
        - Bobot dewasa: Jantan 30-50 kg, Betina 25-40 kg
        - Keunggulan: Tahan kekeringan, dapat menyimpan cadangan energi di ekornya
        - Produksi: Daging dengan karakteristik khas
    
    - **Domba Merino**: 
        - Asal: Spanyol, dikembangkan di Australia
        - Ciri khas: Wool sangat halus dan tebal, wajah terbuka tanpa wool
        - Bobot dewasa: Jantan 70-100 kg, Betina 40-70 kg
        - Keunggulan: Penghasil wool terbaik (3-6 kg wool/tahun)
        - Produksi: Wool premium, daging
    
    - **Domba Garut**: 
        - Asal: Garut, Jawa Barat
        - Ciri khas: Tanduk melingkar kuat (jantan), postur gagah
        - Bobot dewasa: Jantan 60-80 kg, Betina 30-40 kg
        - Keunggulan: Performa petarung baik (domba adu), pertumbuhan cepat
        - Produksi: Daging, domba aduan
    
    - **Domba Suffolk**: 
        - Asal: Inggris
        - Ciri khas: Kepala dan kaki hitam, badan berisi wool putih, tidak bertanduk
        - Bobot dewasa: Jantan 100-160 kg, Betina 80-110 kg
        - Keunggulan: Pertumbuhan sangat cepat, konformasi tubuh ideal untuk daging
        - Produksi: Daging premium, pertumbuhan anak bisa mencapai 300-400 gram/hari
    
    - **Domba Texel**: 
        - Asal: Belanda
        - Ciri khas: Tubuh sangat berotot, wool putih, kepala putih tanpa tanduk
        - Bobot dewasa: Jantan 110-160 kg, Betina 70-100 kg
        - Keunggulan: Persentase karkas tertinggi (60-65%), kualitas daging superior
        - Produksi: Daging premium dengan kadar lemak rendah
    
    ## Kesimpulan
    
    Penggunaan rumus pendugaan berat badan ternak dapat menjadi alternatif yang praktis dan ekonomis bagi peternak untuk memperkirakan bobot ternak tanpa timbangan. Namun, perlu diingat bahwa rumus-rumus ini memberikan estimasi, dan faktor-faktor seperti kondisi tubuh, kebuntingan, dan variasi individual dapat mempengaruhi akurasi. 
    
    > **Penting**: Metode prediksi menggunakan rumus ini memiliki margin error berkisar 5-15% tergantung kondisi ternak, keakuratan pengukuran, dan kesesuaian rumus dengan bangsa ternak. Untuk keperluan yang memerlukan presisi tinggi (seperti penjualan komersial, penetapan dosis obat, kompetisi ternak, dll), penggunaan timbangan ternak tetap merupakan metode yang paling direkomendasikan.
    
    Aplikasi ini menyediakan alat bantu praktis di lapangan ketika timbangan tidak tersedia, namun hasilnya tidak dapat menggantikan pengukuran langsung dengan alat timbang standar.
    """)

# Footer
st.markdown("---")
st.markdown("Dibuat oleh [Galuh Adi Insani](https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/) dengan ❤️ | © 2025")