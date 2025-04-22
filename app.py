import streamlit as st
import pandas as pd
import pickle as pk

# Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.header('Loan Prediction WEBAPP')

# Input fields
no_of_dep = st.slider('Choose No of dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed ?', ['Yes', 'No'])
Annual_Income = st.slider('Choose Annual Income', 0, 1000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 1000000)
Loan_Dur = st.slider('Choose Loan Duration (in years)', 0, 20)
Cibil = st.slider('Choose Cibil Score', 0, 1000)
Assets = st.slider('Choose Assets Value', 0, 1000000)

# Convert categorical inputs to numeric
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

# Prediction
if st.button("Predict"):
    # Make sure column names match exactly as used in training
    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                             columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                      'loan_amount', 'loan_term', 'cibil_score', 'Asserts'])

    # Transform and predict
    pred_data_scaled = scaler.transform(pred_data)
    predict = model.predict(pred_data_scaled)

    # Show result
    if predict[0] == 1:
        st.markdown('✅ **Loan Is Approved**')
    else:
        st.markdown('❌ **Loan Is Rejected**')
