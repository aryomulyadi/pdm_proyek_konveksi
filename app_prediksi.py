import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def format_rupiah(nilai):
    return f"Rp{nilai:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def hitung_rule_based(data):
    jumlah = int(data["jumlah_pesanan"])

    if 1 <= jumlah <= 3:
        harga_dasar = 85000
    elif 4 <= jumlah <= 11:
        harga_dasar = 75000
    elif 12 <= jumlah <= 36:
        harga_dasar = 65000
    elif 37 <= jumlah <= 100:
        harga_dasar = 60000
    else:
        harga_dasar = 55000

    tambahan_bahan = {
        "CC 30S": 0,
        "CC 24S": 9000,
        "CC 20S": 9000,
        "Lacoste Pique": 13000,
        "Lacoste Cotton": 33000,
    }

    tambahan_lengan = {
        "Pendek": 0,
        "Panjang": 10000,
    }

    tambahan_lengan_sablon = {
        "-": 0,
        "A5": 5000,
        "A4": 10000,
    }

    harga_satuan = harga_dasar

    harga_satuan += tambahan_bahan[data["bahan"]]
    harga_satuan += tambahan_lengan[data["model_lengan"]]

    # Sablon depan
    if data["sablon_depan"] == "A3":
        harga_satuan += 5000

    # Sablon belakang
    if data["sablon_belakang"] == "A3":
        harga_satuan += 5000

    # Sablon lengan kiri dan kanan
    harga_satuan += tambahan_lengan_sablon[data["sablon_lengan_kiri"]]
    harga_satuan += tambahan_lengan_sablon[data["sablon_lengan_kanan"]]

    # Sablon bawah
    if data["sablon_bawah"] == "A4":
        harga_satuan += 15000

    total_harga = harga_satuan * jumlah

    return harga_satuan, total_harga


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Prediksi Harga Konveksi",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Prediksi Harga Pesanan Konveksi")
st.write("Form ini digunakan untuk menguji input pesanan baru secara langsung.")

model = load_model()

st.subheader("Input Pesanan Baru")

jumlah_pesanan = st.number_input(
    "Jumlah Pesanan",
    min_value=1,
    value=24,
    step=1
)

bahan = st.selectbox(
    "Bahan",
    ["CC 30S", "CC 24S", "CC 20S", "Lacoste Pique", "Lacoste Cotton"]
)

model_lengan = st.selectbox(
    "Model Lengan",
    ["Pendek", "Panjang"]
)

sablon_depan = st.selectbox(
    "Sablon Depan",
    ["A4", "A3"]
)

sablon_belakang = st.selectbox(
    "Sablon Belakang",
    ["A4", "A3"]
)

sablon_lengan_kiri = st.selectbox(
    "Sablon Lengan Kiri",
    ["-", "A5", "A4"]
)

sablon_lengan_kanan = st.selectbox(
    "Sablon Lengan Kanan",
    ["-", "A5", "A4"]
)

sablon_bawah = st.selectbox(
    "Sablon Bawah",
    ["-", "A4"]
)

input_pesanan = {
    "jumlah_pesanan": jumlah_pesanan,
    "bahan": bahan,
    "model_lengan": model_lengan,
    "sablon_depan": sablon_depan,
    "sablon_belakang": sablon_belakang,
    "sablon_lengan_kiri": sablon_lengan_kiri,
    "sablon_lengan_kanan": sablon_lengan_kanan,
    "sablon_bawah": sablon_bawah,
}

if st.button("Prediksi Harga"):
    input_df = pd.DataFrame([input_pesanan])

    harga_rule_based, total_rule_based = hitung_rule_based(input_pesanan)

    prediksi_ml = float(model.predict(input_df)[0])
    total_ml = prediksi_ml * jumlah_pesanan

    selisih_satuan = abs(harga_rule_based - prediksi_ml)
    selisih_total = abs(total_rule_based - total_ml)

    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Harga Satuan Rule-Based", format_rupiah(harga_rule_based))
        st.metric("Total Rule-Based", format_rupiah(total_rule_based))

    with col2:
        st.metric("Harga Satuan ML", format_rupiah(prediksi_ml))
        st.metric("Total ML", format_rupiah(total_ml))

    st.subheader("Selisih")
    st.write(f"Selisih harga satuan: **{format_rupiah(selisih_satuan)}**")
    st.write(f"Selisih total harga: **{format_rupiah(selisih_total)}**")

    st.info(
        "Harga final sistem tetap menggunakan rule-based. "
        "Prediksi machine learning hanya digunakan sebagai pembanding dan pendukung keputusan."
    )

    st.subheader("Data Input")
    st.dataframe(input_df)