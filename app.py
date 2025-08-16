import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------
# 1. App Title
# --------------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection System")
st.markdown("Detect fraudulent transactions using Machine Learning (Random Forest Classifier).")

# --------------------------------------
# 2. Load Dataset
# --------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Fraud.csv")  # Keep Fraud.csv in same folder as app.py
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------
    # 3. Preprocess Dataset
    # --------------------------------------
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

    if 'isFraud' not in df.columns:
        st.error("Dataset must contain an 'isFraud' column as target variable.")
        st.stop()

    drop_cols = ['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud']
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    y = df['isFraud']

    # --------------------------------------
    # 4. Train-Test Split & Model Training
    # --------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --------------------------------------
    # 5. Model Evaluation
    # --------------------------------------
    st.subheader("📈 Model Evaluation")
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # --------------------------------------
    # 6. Feature Importance
    # --------------------------------------
    st.subheader("🔥 Feature Importance")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # --------------------------------------
    # 7. User Input for Prediction
    # --------------------------------------
    st.subheader("🧮 Try a New Prediction")

    user_input = {}
    for col in X.columns:
        if df[col].dtype in ['float64', 'int64']:
            user_input[col] = st.number_input(f"Enter value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            user_input[col] = st.selectbox(f"Select value for {col}", df[col].unique())

    if st.button("🔍 Predict Fraud"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
else:
    st.stop()
