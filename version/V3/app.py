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
                "reference": "Cole, J.W., et al. (1964). Effects of Type and Breed of British, Zebu and Dairy Cattle on Production. J Animal Science, 23, 115-120."
            },
            "Sapi Peranakan Ongole (PO)": {
                "karkas_percent": {"Jantan": 50.0, "Betina": 47.0},
                "non_karkas_percent": {
                    "Kepala": 7.0, "Kulit": 8.5, "Kaki": 2.5, "Ekor": 0.5,
                    "Darah": 3.5, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 16.5, "Lemak": 5.5
                },
                "meat_percent_of_carcass": 70.0,
                "reference": "Priyanto, R., et al. (1999). Karakteristik Karkas dan Non-Karkas Sapi PO. Media Veteriner, 6(4), 13-17."
            },
            "Sapi Friesian Holstein (FH)": {
                "karkas_percent": {"Jantan": 53.0, "Betina": 48.0},
                "non_karkas_percent": {
                    "Kepala": 6.2, "Kulit": 8.0, "Kaki": 2.2, "Ekor": 0.5,
                    "Darah": 3.3, "Jantung": 0.4, "Hati": 1.4, "Paru-paru": 1.0,
                    "Limpa": 0.3, "Saluran Pencernaan": 15.0, "Lemak": 6.0
                },
                "meat_percent_of_carcass": 72.0,
                "reference": "Purchas, R.W., et al. (2002). Effects of growth potential on tenderness of beef. J Animal Science, 80, 3211-3221."
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

def hitung_berat_badan(lingkar_dada, panjang_badan, jenis_ternak, bangsa, jenis_kelamin):
    """
    Menghitung prediksi berat badan ternak berdasarkan lingkar dada dan panjang badan.
    """
    breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa]
    formula_name = breed_data["formula_name"]
    factor = breed_data["factor"]
    gender_factor = breed_data["gender_factor"][jenis_kelamin]
    
    formula_data = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]
    formula_text = formula_data["formula"]
    calculation_func = formula_data["calculation"]
    
    berat_badan = calculation_func(lingkar_dada, panjang_badan)
    berat_badan = berat_badan * factor * gender_factor
    
    return berat_badan, formula_name, formula_text

def hitung_komponen_karkas(berat_badan, jenis_ternak, bangsa, jenis_kelamin):
    """
    Menghitung komponen karkas dan non-karkas berdasarkan berat badan.
    """
    slaughter_data = SLAUGHTER_DATA[jenis_ternak]["breeds"][bangsa]
    karkas_percent = slaughter_data["karkas_percent"][jenis_kelamin]
    non_karkas_percent = slaughter_data["non_karkas_percent"]
    meat_percent = slaughter_data["meat_percent_of_carcass"]
    
    karkas_weight = (berat_badan * karkas_percent) / 100
    meat_weight = (karkas_weight * meat_percent) / 100
    bone_and_fat_weight = karkas_weight - meat_weight
    
    non_karkas_weights = {}
    for component, percent in non_karkas_percent.items():
        non_karkas_weights[component] = (berat_badan * percent) / 100
    
    return {
        "karkas_percent": karkas_percent,
        "karkas_weight": karkas_weight,
        "meat_weight": meat_weight,
        "meat_percent_of_carcass": meat_percent,
        "meat_percent_of_body": (meat_weight / berat_badan) * 100,
        "bone_and_fat_weight": bone_and_fat_weight,
        "non_karkas_weights": non_karkas_weights,
        "reference": slaughter_data["reference"]
    }

def create_carcass_sankey(karkas_data, berat_badan):
    """Membuat diagram Sankey untuk visualisasi alur karkas."""
    
    # Prepare nodes
    nodes = ["Berat Hidup", "Karkas", "Non-Karkas", "Daging", "Tulang & Lemak"]
    non_karkas_components = list(karkas_data["non_karkas_weights"].keys())
    nodes.extend(non_karkas_components)
    
    # Prepare source, target, and value arrays
    source = []
    target = []
    value = []
    
    # Berat Hidup to Karkas
    source.append(nodes.index("Berat Hidup"))
    target.append(nodes.index("Karkas"))
    value.append(karkas_data["karkas_weight"])
    
    # Berat Hidup to Non-Karkas
    source.append(nodes.index("Berat Hidup"))
    target.append(nodes.index("Non-Karkas"))
    value.append(berat_badan - karkas_data["karkas_weight"])
    
    # Karkas to components
    source.append(nodes.index("Karkas"))
    target.append(nodes.index("Daging"))
    value.append(karkas_data["meat_weight"])
    
    source.append(nodes.index("Karkas"))
    target.append(nodes.index("Tulang & Lemak"))
    value.append(karkas_data["bone_and_fat_weight"])
    
    # Non-Karkas to components
    for component in non_karkas_components:
        source.append(nodes.index("Non-Karkas"))
        target.append(nodes.index(component))
        value.append(karkas_data["non_karkas_weights"][component])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = "blue"
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            color = "rgba(0,0,255,0.2)"
        )
    )])
    
    fig.update_layout(
        title_text="Diagram Alur Karkas dan Non-Karkas",
        font_size=12,
        height=600
    )
    
    return fig

def create_non_karkas_pie(karkas_data):
    """Membuat pie chart untuk komponen non-karkas."""
    
    labels = list(karkas_data["non_karkas_weights"].keys())
    values = list(karkas_data["non_karkas_weights"].values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        hovertemplate="%{label}<br>%{value:.2f} kg<br>%{percent}"
    )])
    
    fig.update_layout(
        title="Distribusi Komponen Non-Karkas",
        height=500
    )
    
    return fig

def create_karkas_distribution_chart(animal_type, breed, gender, weight):
    """Membuat visualisasi distribusi persentase karkas."""
    
    breed_data = SLAUGHTER_DATA[animal_type]["breeds"][breed]
    karkas_percent = breed_data["karkas_percent"][gender]
    
    # Generate data points
    x = np.linspace(karkas_percent - 5, karkas_percent + 5, 100)
    mean = karkas_percent
    std = 1.2
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
    
    # Create visualization
    fig = go.Figure()
    
    # Add distribution curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        fill='tozeroy',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(0, 176, 246, 0.7)', width=2),
        name='Distribusi Persentase Karkas'
    ))
    
    # Add current value marker
    fig.add_trace(go.Scatter(
        x=[karkas_percent],
        y=[0.002],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name=f'Persentase Karkas {breed}'
    ))
    
    # Configure layout
    fig.update_layout(
        title=f"Distribusi Persentase Karkas {breed} ({gender})",
        xaxis_title="Persentase Karkas (%)",
        yaxis_title="Kepadatan Probabilitas",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=400
    )
    
    # Hide y-axis labels
    fig.update_yaxes(showticklabels=False)
    
    return fig

def create_weight_distribution_chart(animal_type, breed, gender, weight):
    """
    Membuat visualisasi distribusi berat badan ternak.
    """
    breed_data = ANIMAL_DATA[animal_type]["breeds"][breed]
    chest_range = breed_data["chest_range"]
    length_range = breed_data["length_range"]
    
    # Calculate expected weight range based on breed characteristics
    if animal_type == "Sapi":
        weight_range = {
            "min": (chest_range["min"] ** 2 * length_range["min"]) / 11000 * breed_data["gender_factor"][gender],
            "max": (chest_range["max"] ** 2 * length_range["max"]) / 11000 * breed_data["gender_factor"][gender]
        }
    elif animal_type == "Kambing":
        weight_range = {
            "min": (chest_range["min"] ** 2 * length_range["min"]) / 20000 * breed_data["gender_factor"][gender],
            "max": (chest_range["max"] ** 2 * length_range["max"]) / 20000 * breed_data["gender_factor"][gender]
        }
    else:  # Domba
        weight_range = {
            "min": (chest_range["min"] ** 2 * length_range["min"]) / 18000 * breed_data["gender_factor"][gender],
            "max": (chest_range["max"] ** 2 * length_range["max"]) / 18000 * breed_data["gender_factor"][gender]
        }
    
    # Generate weight distribution data
    x = np.linspace(weight_range["min"] * 0.8, weight_range["max"] * 1.2, 200)
    mean = (weight_range["min"] + weight_range["max"]) / 2
    std = (weight_range["max"] - weight_range["min"]) / 4
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
    
    # Create visualization
    fig = go.Figure()
    
    # Add distribution area
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        fill='tozeroy',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(0, 176, 246, 0.7)', width=2),
        name='Distribusi Berat'
    ))
    
    # Add normal range area
    fig.add_shape(
        type="rect",
        x0=weight_range["min"],
        y0=0,
        x1=weight_range["max"],
        y1=max(y) * 1.1,
        fillcolor="rgba(0, 255, 0, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Add current weight marker
    fig.add_trace(go.Scatter(
        x=[weight],
        y=[0],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Berat Saat Ini'
    ))
    
    # Add normal range label
    fig.add_annotation(
        x=(weight_range["min"] + weight_range["max"]) / 2,
        y=max(y) * 0.8,
        text=f"Rentang Berat Normal<br>{weight_range['min']:.0f} - {weight_range['max']:.0f} kg",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="green",
        borderwidth=1,
        borderpad=4
    )
    
    # Configure layout
    fig.update_layout(
        title=f"Distribusi Berat Badan {breed} ({gender})",
        xaxis_title="Berat Badan (kg)",
        yaxis_title="Kepadatan Probabilitas",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=400
    )
    
    # Hide y-axis labels since they're not meaningful for density plots
    fig.update_yaxes(showticklabels=False)
    
    return fig

def compare_formulas(animal_type, chest_size, body_length, gender, breed):
    """
    Membandingkan hasil perhitungan dari berbagai rumus yang tersedia.
    """
    breed_data = ANIMAL_DATA[animal_type]["breeds"][breed]
    breed_factor = breed_data["factor"]
    gender_factor = breed_data["gender_factor"][gender]
    
    results = {}
    formulas = ANIMAL_FORMULAS[animal_type]["formulas"]
    
    for formula_name, formula_data in formulas.items():
        # Calculate raw weight
        calculation_func = formula_data["calculation"]
        raw_weight = calculation_func(chest_size, body_length)
        
        # Apply correction factors
        corrected_weight = raw_weight * breed_factor * gender_factor
        
        results[formula_name] = {
            "formula": formula_data["formula"],
            "raw_weight": raw_weight,
            "corrected_weight": corrected_weight,
            "description": formula_data["description"],
            "reference": formula_data["reference"]
        }
    
    return results

def create_breed_comparison_chart(animal_type, chest_size, body_length, gender):
    """
    Membuat visualisasi perbandingan berat antar bangsa ternak.
    """
    breeds = ANIMAL_DATA[animal_type]["breeds"]
    breed_names = []
    weights = []
    formulas = []
    
    # Calculate weight for each breed
    for breed_name, breed_data in breeds.items():
        formula_name = breed_data["formula_name"]
        breed_factor = breed_data["factor"]
        gender_factor = breed_data["gender_factor"][gender]
        
        # Get formula and calculate weight
        formula_data = ANIMAL_FORMULAS[animal_type]["formulas"][formula_name]
        calculation_func = formula_data["calculation"]
        
        weight = calculation_func(chest_size, body_length)
        weight = weight * breed_factor * gender_factor
        
        breed_names.append(breed_name)
        weights.append(weight)
        formulas.append(formula_name)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=breed_names,
            y=weights,
            text=[f"{w:.1f} kg" for w in weights],
            textposition='auto',
            hovertemplate="Bangsa: %{x}<br>" +
                         "Berat: %{y:.1f} kg<br>" +
                         "Rumus: %{customdata}<br>" +
                         "Lingkar Dada: " + str(chest_size) + " cm<br>" +
                         "Panjang Badan: " + str(body_length) + " cm",
            customdata=formulas
        )
    ])
    
    fig.update_layout(
        title=f"Perbandingan Berat {animal_type} {gender}<br>LD={chest_size}cm, PB={body_length}cm",
        xaxis_title="Bangsa",
        yaxis_title="Berat (kg)",
        height=500,
        showlegend=False
    )
    
    return fig

# Main UI components
st.title("🐄 Prediksi Berat Badan Ternak")

st.markdown("""
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

# Show measurement guide
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
        ### 📏 Panduan Pengukuran
        1. **Lingkar Dada (LD)**: 
           - Ukur tepat di belakang bahu (scapula)
           - Lingkarkan pita ukur mengelilingi dada
           - Pastikan pita tidak kendur atau terlalu kencang
        
        2. **Panjang Badan (PB)**:
           - Ukur dari ujung sendi bahu (tuberositas humeri)
           - Sampai ujung tulang duduk (tuber ischii)
           - Gunakan tongkat ukur atau pita ukur yang direntangkan lurus
    """)
with col2:
    st.image("assets/panjangbadan.png", caption="Panduan Pengukuran Panjang dan Lingkar Badan", use_container_width=True)

# Get measurement ranges for selected breed
breed_data = ANIMAL_DATA[jenis_ternak]["breeds"][bangsa_ternak]
chest_range = breed_data["chest_range"]
length_range = breed_data["length_range"]

# Input measurements
lingkar_dada = st.sidebar.number_input(
    "Lingkar Dada (cm)",
    min_value=chest_range["min"] * 0.8,
    max_value=chest_range["max"] * 1.2,
    value=chest_range["min"] + (chest_range["max"] - chest_range["min"]) / 2,
    step=0.5,
    help=f"Ukur lingkar dada ternak dengan pita ukur. Rentang normal untuk {bangsa_ternak}: {chest_range['min']}-{chest_range['max']} cm"
)

panjang_badan = st.sidebar.number_input(
    "Panjang Badan (cm)",
    min_value=length_range["min"] * 0.8,
    max_value=length_range["max"] * 1.2,
    value=length_range["min"] + (length_range["max"] - length_range["min"]) / 2,
    step=0.5,
    help=f"Ukur panjang badan ternak. Rentang normal untuk {bangsa_ternak}: {length_range['min']}-{length_range['max']} cm"
)

# Process when button is clicked
if st.sidebar.button("Hitung Berat Badan", type="primary"):
    # Calculate weight
    berat_badan, formula_name, formula_text = hitung_berat_badan(
        lingkar_dada, panjang_badan, jenis_ternak, bangsa_ternak, jenis_kelamin
    )
    
    # Calculate carcass components
    karkas_data = hitung_komponen_karkas(berat_badan, jenis_ternak, bangsa_ternak, jenis_kelamin)
    
    # Display results in a box
    st.success(f"## Prediksi Berat Badan: **{berat_badan:.2f} kg**")
    
    # Show all visualizations in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Detail Perhitungan",
        "Distribusi Berat",
        "Perbandingan Rumus",
        "Perbandingan Bangsa",
        "Analisis Karkas"
    ])
    
    with tab1:
        # Display calculation details
        st.subheader("Detail Perhitungan")
        
        # Get formula reference
        formula_reference = ANIMAL_FORMULAS[jenis_ternak]["formulas"][formula_name]["reference"]
        
        st.markdown(f"""
        ##### Data Ternak
        - Jenis: **{jenis_ternak}**
        - Bangsa: **{bangsa_ternak}**
        - Jenis Kelamin: **{jenis_kelamin}**
        
        ##### Ukuran Tubuh
        - Lingkar Dada (LD): **{lingkar_dada} cm** (Normal: {chest_range['min']}-{chest_range['max']} cm)
        - Panjang Badan (PB): **{panjang_badan} cm** (Normal: {length_range['min']}-{length_range['max']} cm)
        
        ##### Rumus yang Digunakan
        - Nama Rumus: **{formula_name}**
        - Formula: **{formula_text}**
        - Referensi: *{formula_reference}*
        
        ##### Hasil Perhitungan
        - Berat Badan = **{berat_badan:.2f} kg**
        - Persentase Karkas = **{karkas_data['karkas_percent']:.1f}%**
        - Berat Karkas = **{karkas_data['karkas_weight']:.2f} kg**
        - Berat Daging = **{karkas_data['meat_weight']:.2f} kg**
        """)
        
    with tab2:
        # Weight distribution visualization
        st.plotly_chart(
            create_weight_distribution_chart(jenis_ternak, bangsa_ternak, jenis_kelamin, berat_badan),
            use_container_width=True
        )
        
    with tab3:
        # Formula comparison
        st.subheader("Perbandingan Hasil Antar Rumus")
        formula_results = compare_formulas(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin, bangsa_ternak)
        
        # Create comparison table
        formula_df = pd.DataFrame([
            {
                "Rumus": name,
                "Formula": data["formula"],
                "Berat Dasar (kg)": f"{data['raw_weight']:.2f}",
                "Berat Terkoreksi (kg)": f"{data['corrected_weight']:.2f}",
                "Deskripsi": data["description"]
            }
            for name, data in formula_results.items()
        ])
        
        st.dataframe(formula_df, hide_index=True, use_container_width=True)
        
    with tab4:
        # Breed comparison
        st.plotly_chart(
            create_breed_comparison_chart(jenis_ternak, lingkar_dada, panjang_badan, jenis_kelamin),
            use_container_width=True
        )
        
    with tab5:
        # Karkas analysis
        st.subheader("Analisis Karkas dan Hasil Pemotongan")
        
        # Create three columns
        col1, col2, col3 = st.columns([1.2, 1, 1])
        
        with col1:
            # Main carcass stats
            st.markdown(f"""
            ##### Komponen Utama
            - Berat Hidup: **{berat_badan:.2f} kg**
            - Persentase Karkas: **{karkas_data['karkas_percent']:.1f}%**
            - Berat Karkas: **{karkas_data['karkas_weight']:.2f} kg**
            - Berat Non-Karkas: **{berat_badan - karkas_data['karkas_weight']:.2f} kg**
            
            ##### Komponen Karkas
            - Persentase Daging: **{karkas_data['meat_percent_of_carcass']:.1f}%**
            - Berat Daging: **{karkas_data['meat_weight']:.2f} kg**
            - Berat Tulang & Lemak: **{karkas_data['bone_and_fat_weight']:.2f} kg**
            """)
            
            st.markdown(f"> *Referensi: {karkas_data['reference']}*")
        
        with col2:
            # Sankey diagram
            st.plotly_chart(
                create_carcass_sankey(karkas_data, berat_badan),
                use_container_width=True
            )
            
        with col3:
            # Non-carcass pie chart
            st.plotly_chart(
                create_non_karkas_pie(karkas_data),
                use_container_width=True
            )

# Footer
st.markdown("""
<hr style="height:1px;border:none;color:#333;background-color:#333;margin-top:30px;margin-bottom:20px">
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding:15px; margin-top:10px; margin-bottom:20px">
    <p style="font-size:14px; color:#666">
        Developed by <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank" 
        style="text-decoration:none; color:#0077B5; font-weight:bold">Galuh Adi Insani</a> 
        © {current_year} | All rights reserved
    </p>
</div>
""", unsafe_allow_html=True)