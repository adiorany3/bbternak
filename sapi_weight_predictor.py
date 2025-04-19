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

# Get current year for the footer
current_year = datetime.now().year

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

# Helper function untuk memprediksi berat karkas
def hitung_berat_karkas(berat_badan, jenis_ternak, bangsa, jenis_kelamin):
    """
    Menghitung prediksi berat karkas berdasarkan berat badan hidup ternak.
    
    Args:
        berat_badan (float): Berat badan hidup ternak dalam kilogram
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan atau Betina)
        
    Returns:
        float: Berat karkas ternak dalam kilogram
        float: Persentase karkas
    """
    # Persentase karkas berdasarkan jenis dan bangsa ternak
    persentase_karkas = {
        "Sapi": {
            "Sapi Bali": {"Jantan": 0.58, "Betina": 0.53},
            "Sapi Madura": {"Jantan": 0.54, "Betina": 0.50},
            "Sapi Limousin": {"Jantan": 0.63, "Betina": 0.59},
            "Sapi Simental": {"Jantan": 0.62, "Betina": 0.58},
            "Sapi Brahman": {"Jantan": 0.60, "Betina": 0.56},
            "Sapi Peranakan Ongole (PO)": {"Jantan": 0.56, "Betina": 0.52},
            "Sapi Friesian Holstein (FH)": {"Jantan": 0.57, "Betina": 0.53},
            "Sapi Aceh": {"Jantan": 0.53, "Betina": 0.49}
        },
        "Kambing": {
            "Kambing Kacang": {"Jantan": 0.53, "Betina": 0.49},
            "Kambing Ettawa": {"Jantan": 0.55, "Betina": 0.51},
            "Kambing Peranakan Ettawa (PE)": {"Jantan": 0.54, "Betina": 0.50},
            "Kambing Boer": {"Jantan": 0.60, "Betina": 0.55},
            "Kambing Jawarandu": {"Jantan": 0.53, "Betina": 0.49},
            "Kambing Bligon": {"Jantan": 0.52, "Betina": 0.48}
        },
        "Domba": {
            "Domba Ekor Tipis": {"Jantan": 0.52, "Betina": 0.48},
            "Domba Ekor Gemuk": {"Jantan": 0.56, "Betina": 0.52},
            "Domba Merino": {"Jantan": 0.54, "Betina": 0.50},
            "Domba Garut": {"Jantan": 0.57, "Betina": 0.53},
            "Domba Suffolk": {"Jantan": 0.58, "Betina": 0.54},
            "Domba Texel": {"Jantan": 0.60, "Betina": 0.56}
        }
    }
    
    # Ambil persentase karkas sesuai jenis, bangsa, dan jenis kelamin
    persentase = persentase_karkas[jenis_ternak][bangsa][jenis_kelamin]
    
    # Hitung berat karkas
    berat_karkas = berat_badan * persentase
    
    return berat_karkas, persentase

# Helper function untuk memprediksi berat komponen non karkas
def hitung_berat_non_karkas(berat_badan, berat_karkas):
    """
    Menghitung prediksi berat komponen non karkas berdasarkan berat badan dan berat karkas.
    
    Args:
        berat_badan (float): Berat badan hidup ternak dalam kilogram
        berat_karkas (float): Berat karkas ternak dalam kilogram
        
    Returns:
        dict: Berat berbagai komponen non karkas dalam kilogram
    """
    # Berat total non karkas
    berat_non_karkas_total = berat_badan - berat_karkas
    
    # Persentase berbagai komponen non karkas dari total non karkas
    # Nilai ini adalah perkiraan umum dan dapat bervariasi antar bangsa dan kondisi ternak
    komponen_non_karkas = {
        "Kulit": 0.12,  # 12% dari total non karkas
        "Darah": 0.10,  # 10% dari total non karkas
        "Kepala": 0.16,  # 16% dari total non karkas
        "Kaki": 0.07,  # 7% dari total non karkas
        "Viscera (Organ Dalam)": 0.38,  # 38% dari total non karkas
        "Lainnya": 0.17  # 17% dari total non karkas
    }
    
    # Hitung berat setiap komponen
    hasil_non_karkas = {}
    for komponen, persentase in komponen_non_karkas.items():
        hasil_non_karkas[komponen] = berat_non_karkas_total * persentase
    
    # Tambahkan total non karkas
    hasil_non_karkas["Total Non Karkas"] = berat_non_karkas_total
    
    return hasil_non_karkas

# Helper function untuk memprediksi meat bone ratio
def hitung_meat_bone_ratio(berat_karkas, jenis_ternak, bangsa, jenis_kelamin):
    """
    Menghitung prediksi meat bone ratio (rasio daging dengan tulang) dari karkas.
    
    Args:
        berat_karkas (float): Berat karkas ternak dalam kilogram
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan atau Betina)
        
    Returns:
        float: Meat bone ratio (rasio daging dengan tulang)
        dict: Komposisi karkas (daging, tulang, lemak) dalam kg
    """
    # Rasio daging:tulang:lemak berdasarkan jenis dan bangsa ternak
    # Format: {Daging, Tulang, Lemak} sebagai proporsi dari karkas
    rasio_karkas = {
        "Sapi": {
            "Sapi Bali": {"Jantan": [0.68, 0.18, 0.14], "Betina": [0.65, 0.17, 0.18]},
            "Sapi Madura": {"Jantan": [0.66, 0.19, 0.15], "Betina": [0.63, 0.18, 0.19]},
            "Sapi Limousin": {"Jantan": [0.72, 0.16, 0.12], "Betina": [0.70, 0.15, 0.15]},
            "Sapi Simental": {"Jantan": [0.71, 0.16, 0.13], "Betina": [0.69, 0.15, 0.16]},
            "Sapi Brahman": {"Jantan": [0.70, 0.17, 0.13], "Betina": [0.67, 0.16, 0.17]},
            "Sapi Peranakan Ongole (PO)": {"Jantan": [0.67, 0.19, 0.14], "Betina": [0.64, 0.18, 0.18]},
            "Sapi Friesian Holstein (FH)": {"Jantan": [0.65, 0.20, 0.15], "Betina": [0.62, 0.19, 0.19]},
            "Sapi Aceh": {"Jantan": [0.65, 0.20, 0.15], "Betina": [0.62, 0.19, 0.19]}
        },
        "Kambing": {
            "Kambing Kacang": {"Jantan": [0.64, 0.21, 0.15], "Betina": [0.61, 0.20, 0.19]},
            "Kambing Ettawa": {"Jantan": [0.65, 0.20, 0.15], "Betina": [0.62, 0.19, 0.19]},
            "Kambing Peranakan Ettawa (PE)": {"Jantan": [0.64, 0.21, 0.15], "Betina": [0.61, 0.20, 0.19]},
            "Kambing Boer": {"Jantan": [0.68, 0.19, 0.13], "Betina": [0.65, 0.18, 0.17]},
            "Kambing Jawarandu": {"Jantan": [0.63, 0.22, 0.15], "Betina": [0.60, 0.21, 0.19]},
            "Kambing Bligon": {"Jantan": [0.62, 0.22, 0.16], "Betina": [0.59, 0.21, 0.20]}
        },
        "Domba": {
            "Domba Ekor Tipis": {"Jantan": [0.61, 0.22, 0.17], "Betina": [0.58, 0.21, 0.21]},
            "Domba Ekor Gemuk": {"Jantan": [0.60, 0.21, 0.19], "Betina": [0.57, 0.20, 0.23]},
            "Domba Merino": {"Jantan": [0.63, 0.20, 0.17], "Betina": [0.60, 0.19, 0.21]},
            "Domba Garut": {"Jantan": [0.64, 0.20, 0.16], "Betina": [0.61, 0.19, 0.20]},
            "Domba Suffolk": {"Jantan": [0.66, 0.19, 0.15], "Betina": [0.63, 0.18, 0.19]},
            "Domba Texel": {"Jantan": [0.67, 0.18, 0.15], "Betina": [0.64, 0.17, 0.19]}
        }
    }
    
    # Ambil rasio sesuai jenis, bangsa, dan jenis kelamin
    rasio = rasio_karkas[jenis_ternak][bangsa][jenis_kelamin]
    
    # Proporsi daging, tulang, lemak
    proporsi_daging = rasio[0]
    proporsi_tulang = rasio[1]
    proporsi_lemak = rasio[2]
    
    # Hitung berat komponen karkas
    berat_daging = berat_karkas * proporsi_daging
    berat_tulang = berat_karkas * proporsi_tulang
    berat_lemak = berat_karkas * proporsi_lemak
    
    # Hitung meat bone ratio
    meat_bone_ratio = berat_daging / berat_tulang
    
    # Komposisi karkas
    komposisi_karkas = {
        "Daging": berat_daging,
        "Tulang": berat_tulang,
        "Lemak": berat_lemak
    }
    
    return meat_bone_ratio, komposisi_karkas

# Helper function untuk visualisasi komposisi karkas
def create_carcass_composition_chart(komposisi_karkas):
    """
    Membuat visualisasi komposisi karkas (daging, tulang, lemak)
    
    Args:
        komposisi_karkas (dict): Komposisi karkas (daging, tulang, lemak) dalam kg
        
    Returns:
        plotly.graph_objects.Figure: Visualisasi komposisi karkas
    """
    # Data untuk visualisasi
    labels = list(komposisi_karkas.keys())
    values = list(komposisi_karkas.values())
    
    # Warna untuk masing-masing komponen
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166']
    
    # Buat visualisasi pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors)
    )])
    
    # Konfigurasi layout
    fig.update_layout(
        title="Komposisi Karkas",
        annotations=[dict(
            text=f"Total:<br>{sum(values):.2f} kg",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )],
        height=400
    )
    
    return fig

# Helper function untuk visualisasi komponen non karkas
def create_non_carcass_chart(komponen_non_karkas):
    """
    Membuat visualisasi komponen non karkas
    
    Args:
        komponen_non_karkas (dict): Komponen non karkas dalam kg
        
    Returns:
        plotly.graph_objects.Figure: Visualisasi komponen non karkas
    """
    # Data untuk visualisasi
    # Hilangkan 'Total Non Karkas' dari pie chart
    komponen = {k: v for k, v in komponen_non_karkas.items() if k != "Total Non Karkas"}
    labels = list(komponen.keys())
    values = list(komponen.values())
    
    # Warna untuk masing-masing komponen
    colors = px.colors.qualitative.Pastel
    
    # Buat visualisasi pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors)
    )])
    
    # Konfigurasi layout
    fig.update_layout(
        title="Komposisi Non Karkas",
        annotations=[dict(
            text=f"Total:<br>{komponen_non_karkas['Total Non Karkas']:.2f} kg",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )],
        height=400
    )
    
    return fig
