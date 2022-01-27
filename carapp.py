#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('car-pipeline')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

        from PIL import Image
        image = Image.open('index1.jpg')
        st.image(image,use_column_width=True)
        st.title("Car Price Prediction App")

        Name = st.selectbox('Choose Brand', ['Mercedes-Benz', 'Hyundai','Honda','Audi','Nissan','Toyota','Volkswagen','Tata','Land','Mitsubishi','Renault','Maruthi','BMW','Mahindra','Ford','Porshe','Datsun','Jaguar','Volvo','Chevrolet','Skoda','Mini','Fiat','Jeep','Ambassador','Isuzu','ISUZU','Force','Bentley','Lamborghini'])
        Location = st.selectbox('Location', ['Hyderabad', 'Pune','Chennai','Coimbatore','Mumbai','Jaipur','Kochi','Kolkata','Delhi','Bangalore','Ahmedabad'])
        Kilometers_Driven=st.number_input('Kilometers_Driven', min_value=170, max_value=800000, value=110000)
        Year=st.number_input('Year', min_value=1998, max_value=2019, value=2016)
        Fuel_Type=st.selectbox('Select Fuel type', ['Diesel', 'CNG', 'Petrol', 'LPG'])
        Transmission=st.selectbox('Select Transmission', ['Manual', 'Automatic'])
        Owner_Type=st.selectbox('Select Owner Type', ['First', 'Second','Third','Fourth & Above'])
        Mileage=st.number_input('Enter Mileage', min_value=0.00, max_value=34.00, value=15.00,step=0.01)
        Engine=st.number_input('How much cc engine', min_value=620.00, max_value=6000.00, value=1500.00,step=0.01)
        Power=st.number_input('How much Horse power (bhp)', min_value=34.00, max_value=560.00, value=120.00,step=0.01)
        Seats=st.number_input('How many seats',min_value=2, max_value=10, value=5)
        
     
        output=""

        input_dict = {'Name' : Name, 'Location' : Location,'Year':Year, 'Kilometers_Driven':Kilometers_Driven,
                      'Fuel_Type':Fuel_Type, 'Transmission':Transmission, 'Owner_Type':Owner_Type, 'Mileage':Mileage,
                      'Engine':Engine, 'Power':Power, 'Seats':Seats}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = round(output,2)
            output = str(output)+" Lakhs"
        
        st.info('The Estimated Price of car is :{}'.format(output))
        
        
if __name__ == '__main__':
    run() 
#streamlit run app.py   
    

