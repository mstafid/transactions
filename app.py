import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model yang disimpan
try:
    best_model = joblib.load('best_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Memuat dataset untuk mendapatkan informasi prapemrosesan
df = pd.read_csv('transactions.csv')


# Menyederhanakan fitur numerik
scaler = StandardScaler()
df[['Amount (INR)']] = scaler.fit_transform(df[['Amount (INR)']])

# Mengkodekan UPI ID (karena mereka bisa berupa string)
label_encoder_sender = LabelEncoder()
label_encoder_sender.fit(df['Sender UPI ID'])

label_encoder_receiver = LabelEncoder()
label_encoder_receiver.fit(df['Receiver UPI ID'])

# Fungsi untuk memperluas LabelEncoder dengan label baru
def extend_label_encoder(label_encoder, new_label):
    if new_label not in label_encoder.classes_:
        # Perluas label encoder dengan label baru
        new_classes = np.append(label_encoder.classes_, new_label)
        label_encoder.classes_ = new_classes

# Fungsi prediksi menggunakan model yang disimpan
def predict(sender_upi_id, receiver_upi_id, amount_inr):
    # Periksa dan tambahkan label baru jika perlu
    extend_label_encoder(label_encoder_sender, sender_upi_id)
    extend_label_encoder(label_encoder_receiver, receiver_upi_id)
    
    # Menskalakan input
    scaled_amount = scaler.transform([[amount_inr]])
    # Mengkodekan input
    encoded_sender = label_encoder_sender.transform([sender_upi_id])
    encoded_receiver = label_encoder_receiver.transform([receiver_upi_id])
    # Membuat prediksi
    features = np.array([[encoded_sender[0], encoded_receiver[0], scaled_amount[0][0]]])
    prediction = best_model.predict(features)
    return prediction[0]

# Membuat antarmuka Streamlit
st.title("Prediksi Status Transaksi UPI")
sender_upi_id = st.text_input("Sender UPI ID:")
receiver_upi_id = st.text_input("Receiver UPI ID:")
amount_inr = st.number_input("Amount (INR):", min_value=0.0, step=0.01)

if st.button("Prediksi"):
    try:
        result = predict(sender_upi_id, receiver_upi_id, amount_inr)
        if result == 1:
            st.success("Transaksi Sukses")
        else:
            st.error("Transaksi Gagal")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
