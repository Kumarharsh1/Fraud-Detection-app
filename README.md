# Fraud Detection App

An interactive web application built with **Streamlit** that predicts whether a financial transaction is fraudulent using a trained machine learning model.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-app-padv6u3p7gtqrehhjzv7dn.streamlit.app/)

---

##  Demo

Click the badge above or visit the live demo here:  
https://fraud-detection-app-padv6u3p7gtqrehhjzv7dn.streamlit.app/

---

## Features

- **Interactive UI**: Enter transaction details such as step (time), amount, balances, and transaction type.
- **Model-Based Predictions**: Uses a pre-trained Machine Learning model to classify transactions as *fraudulent* or *legitimate*.
- **One-Hot Encoding for Types**: Supports different transaction types (`CASH_OUT`, `TRANSFER`, `PAYMENT`, `DEBIT`) with `CASH_IN` as the base category.

---

## Folder Structure
fraud-detection-app/
├── streamlit_app.py # Main Streamlit application
├── train_model.py # Script to train and save the ML model
├── fraud_dataset.csv # Dataset used for training (e.g. from PaySim or your source)
├── fraud_model.pkl # Saved trained model (used by the app)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/fraud-detection-app.git
   cd fraud-detection-app
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py

Model Training Script (train_model.py)

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("fraud_dataset.csv")
features = [
    'step', 'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER'
]
target = 'isFraud'

X = df[features]
y = df[target]

# Train and save model
model = RandomForestClassifier()
model.fit(X, y)
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved as fraud_model.pkl")

