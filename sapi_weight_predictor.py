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
        xaxis_title="Bangsa Ternak",
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


def generate_recommendations(berat_badan, lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin):
    """Membuat rekomendasi otomatis berdasarkan hasil prediksi."""
    status, status_note = get_size_status(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak)
    recommendations = [
        f"Status ukuran ternak: {status}. {status_note}",
        "Lakukan pengukuran 2–3 kali, lalu gunakan nilai rata-rata agar prediksi lebih stabil.",
        "Untuk transaksi bernilai besar, tetap gunakan timbangan ternak yang terkalibrasi sebagai pembanding.",
    ]

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
        ("Jenis Ternak", report_data.get("jenis_ternak", "-")),
        ("Bangsa Ternak", report_data.get("bangsa_ternak", "-")),
        ("Jenis Kelamin", report_data.get("jenis_kelamin", "-")),
        ("Lingkar Dada", f"{report_data.get('lingkar_dada', 0):.1f} cm"),
        ("Panjang Badan", f"{report_data.get('panjang_badan', 0):.1f} cm"),
        ("Rumus", report_data.get("formula_name", "-")),
        ("Prediksi Berat Badan", f"{report_data.get('berat_badan', 0):.2f} kg"),
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
            harga_hidup = float(row.get("Harga per Kg", 0) or 0)

            berat, formula_name, formula_text = hitung_berat_badan(ld, pb, jenis, bangsa, kelamin)
            karkas = hitung_komponen_karkas(berat, jenis, bangsa, kelamin)
            status, _ = get_size_status(ld, pb, jenis, bangsa)

            results.append({
                "No": idx + 1,
                "Jenis Ternak": jenis,
                "Bangsa Ternak": bangsa,
                "Jenis Kelamin": kelamin,
                "Lingkar Dada (cm)": ld,
                "Panjang Badan (cm)": pb,
                "Rumus": formula_name,
                "Prediksi Berat (kg)": round(berat, 2),
                "Berat Karkas (kg)": round(karkas["karkas_weight"], 2),
                "Berat Daging (kg)": round(karkas["meat_weight"], 2),
                "Status Ukuran": status,
                "Harga per Kg": harga_hidup,
                "Estimasi Nilai Ternak": round(berat * harga_hidup, 0),
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
                "Berat Karkas (kg)": 0,
                "Berat Daging (kg)": 0,
                "Status Ukuran": "-",
                "Harga per Kg": row.get("Harga per Kg", 0),
                "Estimasi Nilai Ternak": 0,
                "Status Proses": f"Gagal: {exc}",
            })

    return pd.DataFrame(results)

# Judul dan deskripsi aplikasi
st.title("🐄 Prediksi Berat Badan Ternak")

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
        show_image_safe("karkas.jpeg", "Gambar panduan menggunakan file karkas.jpeg.", fallback_paths=["lingkar_dada.png", "version/V3/assets/panjangbadan.png"])

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
        show_image_safe("karkas.jpeg", "Gambar panduan menggunakan file karkas.jpeg.", fallback_paths=["panjang_badan.png", "panjangbadan.png"])

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
# with col2: # Removing this redundant image section
# st.image("panjangbadan.png", caption="Panduan Pengukuran Panjang Badan, ref : https://vetmedicinae.com/cara-menghitung-berat-badan-sapi/", use_container_width=True)

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

st.sidebar.markdown("---")
st.sidebar.subheader("Estimasi Ekonomi")
harga_bobot_hidup = st.sidebar.number_input(
    "Harga per kg bobot hidup (Rp)",
    min_value=0,
    value=65000 if jenis_ternak == "Sapi" else 90000,
    step=1000,
    help="Masukkan harga pasar per kg bobot hidup. Isi 0 jika tidak ingin menghitung nilai ekonomi."
)
harga_karkas = st.sidebar.number_input(
    "Harga per kg karkas (Rp)",
    min_value=0,
    value=0,
    step=1000,
    help="Opsional. Digunakan untuk memperkirakan nilai karkas."
)
harga_daging = st.sidebar.number_input(
    "Harga per kg daging (Rp)",
    min_value=0,
    value=0,
    step=1000,
    help="Opsional. Digunakan untuk memperkirakan nilai daging bersih."
)

# Tombol untuk menghitung berat badan
# st.session_state digunakan agar hasil tidak hilang ketika slider/komponen lain berubah.
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "calculation_history" not in st.session_state:
    st.session_state.calculation_history = []
if "new_calculation" not in st.session_state:
    st.session_state.new_calculation = False

if st.sidebar.button("Hitung Berat Badan", type="primary"):
    st.session_state.show_results = True
    st.session_state.new_calculation = True

if st.session_state.show_results:
    # Add info message to guide users
    st.sidebar.info("👉 Silakan geser ke kanan untuk melihat hasil perhitungan detail, visualisasi data, dan analisis komponen karkas.")
    
    # Hitung berat badan
    berat_badan, formula_name, formula_text = hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)

    if not validate_weight_result(berat_badan, jenis_ternak):
        st.warning(
            "Hasil prediksi terlihat berada di luar rentang wajar. "
            "Periksa kembali satuan pengukuran, rumus yang digunakan, dan data input."
        )
    
    # Hitung komponen karkas
    karkas_data = hitung_komponen_karkas(berat_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)
    
    # Tampilkan hasil dalam kotak
    st.success(f"## Prediksi Berat Badan: **{berat_badan:.2f} kg**")

    # Estimasi nilai ekonomi
    nilai_hidup = berat_badan * harga_bobot_hidup
    nilai_karkas = karkas_data["karkas_weight"] * harga_karkas
    nilai_daging = karkas_data["meat_weight"] * harga_daging
    status_ukuran, status_note = get_size_status(lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak)

    st.subheader("Estimasi Nilai Ekonomi")
    econ_col1, econ_col2, econ_col3 = st.columns(3)
    with econ_col1:
        st.metric("Nilai Bobot Hidup", format_rupiah(nilai_hidup))
        st.caption(f"Harga/kg: {format_rupiah(harga_bobot_hidup)}")
    with econ_col2:
        st.metric("Nilai Karkas", format_rupiah(nilai_karkas))
        st.caption(f"Harga/kg: {format_rupiah(harga_karkas)}")
    with econ_col3:
        st.metric("Nilai Daging", format_rupiah(nilai_daging))
        st.caption(f"Harga/kg: {format_rupiah(harga_daging)}")

    st.info(f"**Status ukuran:** {status_ukuran}. {status_note}")

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
            "Harga/kg Bobot Hidup": harga_bobot_hidup,
            "Estimasi Nilai Ternak": round(nilai_hidup, 0),
        })
        st.session_state.new_calculation = False
    
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

    st.subheader("Rekomendasi Otomatis")
    for recommendation in generate_recommendations(berat_badan, lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin):
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
        "harga_hidup": harga_bobot_hidup,
        "nilai_hidup": nilai_hidup,
        "harga_karkas": harga_karkas,
        "nilai_karkas": nilai_karkas,
        "harga_daging": harga_daging,
        "nilai_daging": nilai_daging,
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
st.subheader("Riwayat Perhitungan dan Mode Banyak Ternak")
riwayat_tab, batch_tab = st.tabs(["📋 Riwayat Perhitungan", "📤 Upload Banyak Ternak"])

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

    Kolom wajib:
    - `Jenis Ternak`
    - `Bangsa Ternak`
    - `Jenis Kelamin`
    - `Lingkar Dada`
    - `Panjang Badan`

    Kolom opsional:
    - `Harga per Kg`
    """)

    template_df = pd.DataFrame({
        "Jenis Ternak": ["Sapi", "Kambing", "Domba"],
        "Bangsa Ternak": ["Sapi Bali", "Kambing Kacang", "Domba Garut"],
        "Jenis Kelamin": ["Jantan", "Betina", "Jantan"],
        "Lingkar Dada": [175, 65, 80],
        "Panjang Badan": [150, 55, 70],
        "Harga per Kg": [65000, 90000, 90000],
    })

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

# Footer with LinkedIn profile link and improved styling
st.markdown("""
<hr style="height:1px;border:none;color:#333;background-color:#333;margin-top:30px;margin-bottom:20px">
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding:15px; margin-top:10px; margin-bottom:20px">
    <p style="font-size:16px; color:#555">
        &copy; {current_year} Developed by: 
        <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank" 
           style="text-decoration:none; color:#0077B5; font-weight:bold">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" 
                 width="16" height="16" style="vertical-align:middle; margin-right:5px">
            Galuh Adi Insani
        </a> 
        with <span style="color:#e25555">❤️</span>
    </p>
    <p style="font-size:12px; color:#777">All rights reserved.</p>
</div>
""", unsafe_allow_html=True)