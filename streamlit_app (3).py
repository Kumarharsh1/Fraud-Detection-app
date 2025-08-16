import streamlit as st
import pandas as pd
import pickle
import os

# Check if the model file exists, otherwise raise an error
if not os.path.exists('fraud_model.pkl'):
    st.error("Error: The 'fraud_model.pkl' file was not found. Please train and save your model first.")
else:
    # Load the saved model
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)

# Title and description for your app
st.title('Fraud Transaction Prediction App')
st.write('This app predicts whether a transaction is fraudulent based on a few key inputs.')

# Input fields for the user to enter data
st.header('Transaction Details')
step = st.number_input('Step (Time in hours)', min_value=0, max_value=744, value=1)
amount = st.number_input('Amount', min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input('Old Balance of Origin Account', min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input('New Balance of Origin Account', min_value=0.0, value=900.0)
oldbalanceDest = st.number_input('Old Balance of Destination Account', min_value=0.0, value=500.0)
newbalanceDest = st.number_input('New Balance of Destination Account', min_value=0.0, value=600.0)
transaction_type = st.selectbox('Transaction Type', ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])

# Prepare the input data for the model
def prepare_data(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type):
    # This dictionary must match the feature set your model was trained on.
    data = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_CASH_OUT': 0, 'type_DEBIT': 0, 'type_PAYMENT': 0, 'type_TRANSFER': 0
    }
    
    # Set the appropriate one-hot encoded column to 1
    # Note: 'CASH_IN' is the base category, so its dummy variable is not in the data dictionary.
    if transaction_type != 'CASH_IN':
        column_name = f'type_{transaction_type}'
        if column_name in data:
            data[column_name] = 1

    df_predict = pd.DataFrame([data])
    return df_predict

# Make prediction when the button is clicked
if st.button('Predict'):
    input_data = prepare_data(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type)
    
    try:
        prediction = model.predict(input_data)[0]
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error('🚫 The transaction is predicted to be FRAUDULENT.')
        else:
            st.success('✅ The transaction is predicted to be LEGITIMATE.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")