import os
import json
import joblib
import pandas as pd


# ============================================================
# PATH
# ============================================================

BEST_MODEL_PATH = "models/best_model.pkl"
OUTPUT_PATH = "metrics/single_prediction_result.json"

os.makedirs("metrics", exist_ok=True)


# ============================================================
# FUNGSI RULE-BASED PRICE
# ============================================================

def hitung_harga_rule_based(
    jumlah_pesanan,
    bahan,
    model_lengan,
    sablon_depan,
    sablon_belakang,
    sablon_lengan_kiri,
    sablon_lengan_kanan,
    sablon_bawah
):
    """
    Menghitung harga satuan kaos berdasarkan aturan harga konveksi.
    Fungsi ini menjadi acuan utama harga final pada sistem web.
    """

    # Harga dasar berdasarkan jumlah pesanan
    if 1 <= jumlah_pesanan <= 3:
        harga_dasar = 85000
    elif 4 <= jumlah_pesanan <= 11:
        harga_dasar = 75000
    elif 12 <= jumlah_pesanan <= 36:
        harga_dasar = 65000
    elif 37 <= jumlah_pesanan <= 100:
        harga_dasar = 60000
    elif jumlah_pesanan > 100:
        harga_dasar = 55000
    else:
        raise ValueError("Jumlah pesanan harus lebih dari 0.")

    # Tambahan harga bahan
    tambahan_bahan = {
        "CC 30S": 0,
        "CC 24S": 9000,
        "CC 20S": 9000,
        "Lacoste Pique": 13000,
        "Lacoste Cotton": 33000
    }

    # Tambahan model lengan
    tambahan_lengan = {
        "Pendek": 0,
        "Panjang": 10000
    }

    # Tambahan sablon depan
    tambahan_sablon_depan = {
        "A4": 0,
        "A3": 5000
    }

    # Tambahan sablon belakang
    tambahan_sablon_belakang = {
        "A4": 0,
        "A3": 5000
    }

    # Tambahan sablon lengan kiri
    tambahan_sablon_lengan_kiri = {
        "-": 0,
        "A5": 5000,
        "A4": 10000
    }

    # Tambahan sablon lengan kanan
    tambahan_sablon_lengan_kanan = {
        "-": 0,
        "A5": 5000,
        "A4": 10000
    }

    # Tambahan sablon bawah
    tambahan_sablon_bawah = {
        "-": 0,
        "A4": 15000
    }

    try:
        harga_satuan = (
            harga_dasar
            + tambahan_bahan[bahan]
            + tambahan_lengan[model_lengan]
            + tambahan_sablon_depan[sablon_depan]
            + tambahan_sablon_belakang[sablon_belakang]
            + tambahan_sablon_lengan_kiri[sablon_lengan_kiri]
            + tambahan_sablon_lengan_kanan[sablon_lengan_kanan]
            + tambahan_sablon_bawah[sablon_bawah]
        )
    except KeyError as error:
        raise ValueError(f"Input tidak valid pada nilai: {error}")

    total_harga = harga_satuan * jumlah_pesanan

    return harga_satuan, total_harga


# ============================================================
# DATA INPUT PESANAN BARU
# ============================================================

data_pesanan = {
    "jumlah_pesanan": 24,
    "bahan": "CC 24S",
    "model_lengan": "Panjang",
    "sablon_depan": "A3",
    "sablon_belakang": "A4",
    "sablon_lengan_kiri": "A5",
    "sablon_lengan_kanan": "-",
    "sablon_bawah": "A4"
}


# ============================================================
# LOAD MODEL TERBAIK
# ============================================================

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        "File best_model.pkl tidak ditemukan. "
        "Jalankan terlebih dahulu: python src/modeling.py"
    )

model = joblib.load(BEST_MODEL_PATH)


# ============================================================
# PREDIKSI MODEL MACHINE LEARNING
# ============================================================

input_df = pd.DataFrame([data_pesanan])

prediksi_ml = model.predict(input_df)[0]
prediksi_ml = round(float(prediksi_ml), 2)

total_prediksi_ml = prediksi_ml * data_pesanan["jumlah_pesanan"]


# ============================================================
# HITUNG HARGA RULE-BASED
# ============================================================

harga_rule_based, total_rule_based = hitung_harga_rule_based(
    jumlah_pesanan=data_pesanan["jumlah_pesanan"],
    bahan=data_pesanan["bahan"],
    model_lengan=data_pesanan["model_lengan"],
    sablon_depan=data_pesanan["sablon_depan"],
    sablon_belakang=data_pesanan["sablon_belakang"],
    sablon_lengan_kiri=data_pesanan["sablon_lengan_kiri"],
    sablon_lengan_kanan=data_pesanan["sablon_lengan_kanan"],
    sablon_bawah=data_pesanan["sablon_bawah"]
)


# ============================================================
# HITUNG SELISIH
# ============================================================

selisih_harga_satuan = abs(harga_rule_based - prediksi_ml)
selisih_total_harga = abs(total_rule_based - total_prediksi_ml)


# ============================================================
# SIMPAN HASIL
# ============================================================

hasil_prediksi = {
    "input_pesanan": data_pesanan,
    "hasil_rule_based": {
        "harga_satuan": harga_rule_based,
        "total_harga": total_rule_based
    },
    "hasil_machine_learning": {
        "model": "best_model.pkl",
        "prediksi_harga_satuan": prediksi_ml,
        "prediksi_total_harga": round(total_prediksi_ml, 2)
    },
    "perbandingan": {
        "selisih_harga_satuan": round(selisih_harga_satuan, 2),
        "selisih_total_harga": round(selisih_total_harga, 2)
    }
}

with open(OUTPUT_PATH, "w") as file:
    json.dump(hasil_prediksi, file, indent=4)


# ============================================================
# OUTPUT TERMINAL
# ============================================================

print("\nHASIL PREDIKSI PESANAN BARU")
print("=" * 70)

print("\nInput Pesanan:")
for key, value in data_pesanan.items():
    print(f"{key}: {value}")

print("\nHasil Rule-Based:")
print(f"Harga satuan : Rp{harga_rule_based:,.0f}")
print(f"Total harga  : Rp{total_rule_based:,.0f}")

print("\nHasil Machine Learning:")
print(f"Prediksi harga satuan : Rp{prediksi_ml:,.2f}")
print(f"Prediksi total harga  : Rp{total_prediksi_ml:,.2f}")

print("\nPerbandingan:")
print(f"Selisih harga satuan : Rp{selisih_harga_satuan:,.2f}")
print(f"Selisih total harga  : Rp{selisih_total_harga:,.2f}")

print("\nHasil disimpan di:")
print(OUTPUT_PATH)

print("=" * 70)