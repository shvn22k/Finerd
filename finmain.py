import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import plotly.graph_objects as go

model = pickle.load(open('american-loan-approval.pkl', 'rb'))

st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select an option", ("Loan Approval Prediction", "Mutual Funds Insights"))

if menu == "Loan Approval Prediction":
    st.image("LOAN APPROVAL PREDICTION.png", caption = 'Model is made on american loan approval dataset, so please enter values in dollars.' , use_column_width=True)
    
    
    no_of_dependents = st.number_input("Enter number of dependents", min_value=0, max_value=10, value=2)
    education_status = st.selectbox("Educational Status", [" Not Graduate", " Graduate"])
    self_employed = st.radio("Are you self employed?", [' Yes', ' No'])
    annual_income = st.number_input("Annual Income", min_value=10000)
    loan_amount = st.number_input("Enter loan amount", min_value=10000)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=40)
    credit_score = st.number_input("Credit/CIBIL Score", min_value=300, max_value=850, value=700)
    residential_assets_value = st.number_input("Residential assets value", min_value=1000000)
    commercial_assets_value = st.number_input("Commercial assets value", min_value=1000000)
    luxury_assets_value = st.number_input("Luxury assets value", min_value=1000000)
    bank_assets_value = st.number_input("Bank assets value", min_value=1000000)
    
    inputdata = {' no_of_dependents': no_of_dependents,
                 ' education': education_status,
                 ' self_employed': self_employed,
                 ' income_annum': annual_income,
                 ' loan_amount': loan_amount,
                 ' loan_term': loan_term,
                 ' cibil_score': credit_score,
                 ' residential_assets_value': residential_assets_value,
                 ' commercial_assets_value': commercial_assets_value,
                 ' luxury_assets_value': luxury_assets_value,
                 ' bank_asset_value': bank_assets_value}


    

    if st.button("Predict Loan Approval"):
        inputdata = pd.DataFrame([inputdata])
        prediction = model.predict(inputdata)
        if prediction[0] == 1:
            st.success("High chances of Loan Approval.")
        elif prediction[0] == 0:
            st.error("Low chances of Loan Approval.")
        else:
            st.error("An error occured.")

elif menu == "Mutual Funds Insights":
    st.image("mutualfs.png")
    url = 'https://api.mfapi.in/mf'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
    else:
        st.write("Error:", response.status_code)

    available_funds = pd.DataFrame(data)

    fundlist = list(available_funds['schemeName'])

    fund_name = st.selectbox("Select Mutual Fund", fundlist)
    sid = available_funds.iloc[fundlist.index(fund_name)]['schemeCode']
    schemeDataUrl = f"https://api.mfapi.in/mf/{sid}"

    response2 = requests.get(schemeDataUrl)

    if response2.status_code == 200:
        histData = response2.json()
    else:
        st.write("Error:", response2.status_code)

    df = pd.DataFrame(histData['data'], columns=['date', 'nav'])
    schemeHistData = df

    schemeHistData['date'] = pd.to_datetime(schemeHistData['date'], format='%d-%m-%Y') 
    schemeHistData['nav'] = pd.to_numeric(schemeHistData['nav'])

    if st.button("Show Insights"):
        st.write(f"Displaying insights for {fund_name}")
        schemeHistData['date'] = pd.to_datetime(schemeHistData['date'])
        schemeHistData.set_index('date', inplace=True)


        fig = go.Figure()


        fig.add_trace(go.Scatter(x=schemeHistData.index, 
                                y=schemeHistData['nav'], 
                                mode='lines', 
                                name='NAV'))


        x = np.arange(len(schemeHistData['nav']))
        z = np.polyfit(x, schemeHistData['nav'], 1)
        p = np.poly1d(z)

        fig.add_trace(go.Scatter(x=schemeHistData.index, 
                                y=p(x), 
                                mode='lines', 
                                name='Trend Line', 
                                line=dict(dash='dash', color='red')))


        fig.update_layout(
            title='Scheme NAV with Trend Line',
            xaxis_title='Date',
            yaxis_title='NAV',
            legend=dict(x=0, y=1),
            template='plotly'
        )


        st.plotly_chart(fig)



