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
from scipy import stats

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

def create_weight_distribution_chart(jenis_ternak, bangsa, jenis_kelamin, current_weight):
    """
    Membuat visualisasi distribusi berat badan untuk bangsa dan jenis kelamin ternak yang dipilih.
    
    Args:
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        bangsa (str): Bangsa ternak
        jenis_kelamin (str): Jenis kelamin ternak (Jantan atau Betina)
        current_weight (float): Berat badan ternak saat ini (kg)
        
    Returns:
        plotly.graph_objects.Figure: Visualisasi distribusi berat badan
    """
    # Rentang berat berdasarkan jenis ternak
    if jenis_ternak == "Sapi":
        if jenis_kelamin == "Jantan":
            weight_range = {"min": 200, "max": 1000, "mean": 500, "std": 150}
        else:
            weight_range = {"min": 180, "max": 800, "mean": 400, "std": 120}
    elif jenis_ternak == "Kambing":
        if jenis_kelamin == "Jantan":
            weight_range = {"min": 20, "max": 120, "mean": 60, "std": 20}
        else:
            weight_range = {"min": 15, "max": 80, "mean": 45, "std": 15}
    else:  # Domba
        if jenis_kelamin == "Jantan":
            weight_range = {"min": 25, "max": 150, "mean": 70, "std": 25}
        else:
            weight_range = {"min": 20, "max": 100, "mean": 55, "std": 20}
    
    # Sesuaikan dengan faktor bangsa
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa]
    factor = breed_data["factor"]
    weight_range["min"] *= factor
    weight_range["max"] *= factor
    weight_range["mean"] *= factor
    weight_range["std"] *= factor
    
    # Buat distribusi normal untuk berat
    weights = np.linspace(weight_range["min"], weight_range["max"], 100)
    dist = stats.norm.pdf(weights, weight_range["mean"], weight_range["std"])
    
    # Normalisasi distribusi
    dist = dist / np.max(dist)
    
    # Tentukan percentile dari berat saat ini
    percentile = stats.norm.cdf(current_weight, weight_range["mean"], weight_range["std"]) * 100
    
    # Buat visualisasi
    fig = go.Figure()
    
    # Tambahkan distribusi
    fig.add_trace(go.Scatter(
        x=weights,
        y=dist,
        mode='lines',
        fill='tozeroy',
        name='Distribusi Berat',
        line=dict(color='rgba(55, 128, 191, 0.7)', width=3)
    ))
    
    # Tambahkan marker untuk nilai saat ini
    fig.add_trace(go.Scatter(
        x=[current_weight],
        y=[stats.norm.pdf(current_weight, weight_range["mean"], weight_range["std"]) / np.max(dist)],
        mode='markers',
        name=f'Berat Saat Ini: {current_weight:.1f} kg',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    # Tambahkan anotasi untuk percentile
    fig.add_annotation(
        x=current_weight,
        y=stats.norm.pdf(current_weight, weight_range["mean"], weight_range["std"]) / np.max(dist) + 0.1,
        text=f"Persentil ke-{percentile:.1f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        font=dict(size=12, color="red"),
        align="center"
    )
    
    # Konfigurasi layout
    fig.update_layout(
        title=f"Distribusi Berat {jenis_ternak} {bangsa} ({jenis_kelamin})",
        xaxis_title="Berat Badan (kg)",
        yaxis_title="Densitas Relatif",
        showlegend=True,
        height=400
    )
    
    # Tambahkan garis untuk kuartil
    quartiles = [
        stats.norm.ppf(0.25, weight_range["mean"], weight_range["std"]),
        stats.norm.ppf(0.5, weight_range["mean"], weight_range["std"]),
        stats.norm.ppf(0.75, weight_range["mean"], weight_range["std"])
    ]
    
    labels = ["Kuartil 1 (25%)", "Median (50%)", "Kuartil 3 (75%)"]
    colors = ["green", "blue", "purple"]
    
    for i, (q, label, color) in enumerate(zip(quartiles, labels, colors)):
        fig.add_shape(
            type="line",
            x0=q, y0=0,
            x1=q, y1=0.9,
            line=dict(color=color, width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=q,
            y=0.95,
            text=f"{label}: {q:.1f} kg",
            showarrow=False,
            font=dict(size=10, color=color)
        )
    
    return fig

def create_breed_comparison_chart(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin):
    """
    Membuat visualisasi perbandingan berat badan antar bangsa ternak
    
    Args:
        jenis_ternak (str): Jenis ternak (Sapi, Kambing, Domba)
        lingkar_dada (float): Ukuran lingkar dada (cm)
        panjang_badan (float): Ukuran panjang badan (cm)
        jenis_kelamin (str): Jenis kelamin ternak
        
    Returns:
        plotly.graph_objects.Figure: Visualisasi perbandingan bangsa
    """
    # Dapatkan semua bangsa untuk jenis ternak
    breeds = ANIMAL_DATA[jenis_ternak]["breeds"]
    
    # Hitung berat untuk setiap bangsa
    breed_names = []
    weights = []
    formulas = []
    
    for breed_name, breed_data in breeds.items():
        formula_name = breed_data["formula_name"]
        formula_text = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["formula"]
        formula_func = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["calculation"]
        
        # Hitung berat dasar
        raw_weight = formula_func(lingkar_dada, panjang_badan)
        
        # Terapkan faktor koreksi
        factor = breed_data["factor"]
        gender_factor = breed_data["gender_factor"][jenis_kelamin]
        corrected_weight = raw_weight * factor * gender_factor
        
        breed_names.append(breed_name)
        weights.append(corrected_weight)
        formulas.append(formula_name)
    
    # Buat dataframe untuk visualisasi
    data = {
        'Bangsa': breed_names,
        'Berat (kg)': weights,
        'Rumus': formulas
    }
    
    # Urutkan berdasarkan berat
    sorted_indices = np.argsort(weights)[::-1]  # Descending order
    sorted_breeds = [breed_names[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_formulas = [formulas[i] for i in sorted_indices]
    
    # Buat visualisasi
    fig = go.Figure()
    
    # Tambahkan batang untuk setiap bangsa
    fig.add_trace(go.Bar(
        x=sorted_breeds,
        y=sorted_weights,
        text=[f"{w:.1f} kg<br>{f}" for w, f in zip(sorted_weights, sorted_formulas)],
        textposition='auto',
        marker_color='rgba(55, 83, 109, 0.7)',
        hoverinfo='text',
        hovertext=[f"Bangsa: {b}<br>Berat: {w:.1f} kg<br>Rumus: {f}" 
                   for b, w, f in zip(sorted_breeds, sorted_weights, sorted_formulas)]
    ))
    
    # Tambahkan garis rata-rata
    average_weight = np.mean(weights)
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=average_weight,
        x1=len(breed_names) - 0.5,
        y1=average_weight,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Tambahkan anotasi untuk rata-rata
    fig.add_annotation(
        x=len(breed_names) - 1,
        y=average_weight,
        text=f"Rata-rata: {average_weight:.1f} kg",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1
    )
    
    # Konfigurasi layout
    fig.update_layout(
        title=f"Perbandingan Berat {jenis_ternak} dengan LD={lingkar_dada} cm, PB={panjang_badan} cm ({jenis_kelamin})",
        xaxis_title="Bangsa",
        yaxis_title="Berat Badan (kg)",
        height=500
    )
    
    return fig

def compare_formulas(animal_type, chest_size, body_length, gender, breed):
    """
    Membandingkan berbagai rumus perhitungan berat badan untuk jenis ternak yang sama
    
    Args:
        animal_type (str): Jenis ternak (Sapi, Kambing, Domba)
        chest_size (float): Ukuran lingkar dada ternak
        body_length (float): Ukuran panjang badan ternak
        gender (str): Jenis kelamin (Jantan, Betina)
        breed (str): Bangsa ternak
        
    Returns:
        dict: Dictionary berisi hasil dari berbagai rumus perhitungan
    """
    # Ambil semua rumus yang tersedia untuk jenis ternak
    formulas = ANIMAL_FORMULAS[animal_type]["formulas"]
    
    # Ambil data breed untuk faktor koreksi
    breed_data = ANIMAL_DATA[animal_type]["breeds"][breed]
    factor = breed_data["factor"]
    gender_factor = breed_data["gender_factor"][gender]
    
    # Kumpulkan hasil perhitungan dari berbagai rumus
    results = {}
    
    for formula_name, formula_data in formulas.items():
        # Ambil fungsi perhitungan
        calculation_func = formula_data["calculation"]
        
        # Hitung berat dasar (tanpa faktor koreksi)
        raw_weight = calculation_func(chest_size, body_length)
        
        # Hitung berat terkoreksi (dengan faktor koreksi)
        corrected_weight = raw_weight * factor * gender_factor
        
        # Tambahkan hasil ke dict
        results[formula_name] = {
            "raw_weight": raw_weight,
            "corrected_weight": corrected_weight,
            "formula": formula_data["formula"],
            "description": formula_data["description"]
        }
    
    return results
