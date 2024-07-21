import streamlit as st
import numpy as np
import joblib

# Memuat model
model = joblib.load('best_model.pkl')

# Judul aplikasi
st.title("Prediksi Status Transaksi")

# Membuat input form
amount = st.number_input("Amount (INR)", min_value=0.0, value=0.0)
# Tambahkan input sesuai fitur yang digunakan

# Tombol prediksi
if st.button("Predict"):
    data = np.array([[amount]])  # Sesuaikan dengan jumlah fitur
    prediction = model.predict(data)
    output = "Berhasil" if prediction[0] == 1 else "Gagal"
    st.write(f"Status Transaksi: {output}")
