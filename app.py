import streamlit as st
import pandas as pd
import joblib
import requests
import os

# Fungsi untuk memuat data CSV
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/AndikaBN/predict_price_pc/refs/heads/main/laptop_data.csv')

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1ktcqRauXebQCm32e77KYAdSMc4gJTiBb'
    model_path = 'laptop_model.pkl'

    # Unduh model jika belum ada
    if not os.path.exists(model_path):
        st.write('Downloading model...')
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write('Model downloaded successfully.')

    # Muat model
    st.write(f'Loading model from {model_path}...')
    try:
        model = joblib.load(model_path)
        st.write('Model loaded successfully.')
    except Exception as e:
        st.error(f'Error loading model: {e}')
        raise
    return model

# Fungsi utama aplikasi
def main():
    st.title('Laptop Price Prediction')

    # Muat data
    data = load_data()

    st.sidebar.header('Input Features')

    # Fungsi untuk mengatur input dari user
    def user_input_features():
        features = {}
        for col in data.columns[:-1]:
            if data[col].dtype == 'object': 
                features[col] = st.sidebar.selectbox(f'Select {col}', data[col].unique())
            else:  
                features[col] = st.sidebar.number_input(
                    f'Enter {col}',
                    min_value=float(data[col].min()),
                    max_value=float(data[col].max())
                )
        return pd.DataFrame(features, index=[0])

    # Ambil input dari user
    input_df = user_input_features()

    st.write('### Input Features')
    st.write(input_df)

    # Muat model
    model = load_model()

    # Prediksi harga laptop
    if st.button('Predict'):
        try:
            # Pastikan input_df hanya berisi kolom fitur yang diharapkan oleh model
            prediction = model.predict(input_df)
            formatted_prediction = f'$ {prediction[0]:,.2f}'
            st.write('### Predicted Laptop Price')
            st.write(formatted_prediction)
        except Exception as e:
            st.error(f'Error making prediction: {e}')

# Jalankan aplikasi
if __name__ == '__main__':
    main()
