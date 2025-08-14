import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv("fraud_dataset.csv")  # replace with your dataset name

# Keep only the columns your Streamlit app expects
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
target = 'isFraud'

X = df[features]
y = df[target]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as fraud_model.pkl")
