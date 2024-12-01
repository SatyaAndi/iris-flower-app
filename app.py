import streamlit as st
import numpy as np
import joblib

# Memuat model
model = joblib.load('iris_model.pkl')

# Judul aplikasi
st.title("Prediksi Jenis Bunga Iris")

# Penjelasan aplikasi
st.write("""
Aplikasi ini memprediksi jenis bunga Iris berdasarkan data yang Anda masukkan.
Model ini menggunakan Logistic Regression yang telah dilatih sebelumnya.
""")

# Input dari pengguna
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Tombol untuk memproses prediksi
if st.button("Prediksi"):
    if sepal_length > 0 and sepal_width > 0 and petal_length > 0 and petal_width > 0:
        # Data input pengguna
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Melakukan prediksi
        prediction = model.predict(input_data)
        species = ['Setosa', 'Versicolor', 'Virginica']
        st.success(f"Jenis bunga Iris yang diprediksi adalah: {species[prediction[0]]}")
    else:
        st.warning("Harap masukkan semua nilai untuk melakukan prediksi.")
