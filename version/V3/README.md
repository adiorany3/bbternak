# Livestock Weight Prediction Application

A Streamlit web application for predicting livestock weight based on body measurements using various formulas specific to different breeds and types of livestock.

## Features
- Weight prediction for cattle, goats, and sheep
- Support for multiple breeds with breed-specific formulas
- Visualization of weight distribution and comparisons
- Carcass weight prediction and analysis
- Interactive data visualization

## Setup
1. Ensure Python 3.8+ is installed
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
```bash
streamlit run app.py
```

## Project Structure
```
.
├── .github/
│   └── copilot-instructions.md
├── assets/
│   └── panjangbadan.png
├── app.py
├── requirements.txt
└── README.md
```

## Data Sources and References
The application uses formulas and data from various scientific publications and livestock research institutions. All references are cited within the application.