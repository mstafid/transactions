import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('transactions.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Sender Name', 'Receiver Name', 'Amount (INR)', 'status']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Amount (INR)']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                processed_input[column] = label_encoders[column].transform(['Unknown'])
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

st.title('Prediksi Status Transaksi')

st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    </style>
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Input user
sender_name = st.text_input('Nama Pengirim')
receiver_name = st.text_input('Nama Penerima')
amount = st.number_input('Jumlah (INR)', min_value=0)
status = st.selectbox('Status', ['Sukses', 'Gagal'])

user_input = {
    'Sender Name': sender_name,
    'Receiver Name': receiver_name,
    'Amount (INR)': amount,
    'Status': status
}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.write(f'Prediction: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

 # Tambahkan elemen HTML untuk output
st.markdown("""
    <h3>Output Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
