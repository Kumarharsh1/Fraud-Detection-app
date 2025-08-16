import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Fraud.csv")

# One-hot encode the 'type' column
df = pd.get_dummies(df, columns=['type'], drop_first=False)

# Define features (after one-hot encoding)
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest',
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
target = 'isFraud'

# Keep only rows where target exists
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as fraud_model.pkl")
