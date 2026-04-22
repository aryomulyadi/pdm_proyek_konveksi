import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/banjar_custom_penjualan.xlsx"
SHEET_NAME = "Data Penjualan"

TRAIN_PATH = "data/prepared/train.csv"
TEST_PATH = "data/prepared/test.csv"
PLOT_PATH = "plots/data_distribution.png"
SUMMARY_PATH = "data/prepared/data_summary.json"


def main():
    # buat folder output jika belum ada
    os.makedirs("data/prepared", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # baca data
    df = pd.read_excel(RAW_PATH, sheet_name=SHEET_NAME)

    # cleaning ringan
    kolom_sablon_opsional = [
        "sablon_lengan_kiri",
        "sablon_lengan_kanan",
        "sablon_bawah"
    ]

    for col in kolom_sablon_opsional:
        if col in df.columns:
            df[col] = df[col].fillna("-")

    if "tanggal_order" in df.columns:
        df["tanggal_order"] = pd.to_datetime(df["tanggal_order"], errors="coerce")

    # fitur dan target
    fitur = [
        "jumlah_pesanan",
        "bahan",
        "model_lengan",
        "sablon_depan",
        "sablon_belakang",
        "sablon_lengan_kiri",
        "sablon_lengan_kanan",
        "sablon_bawah"
    ]
    target = "harga_satuan"
    stratify_col = "kategori_jumlah"

    # validasi kolom wajib
    required_cols = fitur + [target, stratify_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")

    df_model = df[required_cols].copy()

    # split train-test
    X = df_model[fitur]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=df_model[stratify_col]
    )

    # gabungkan kembali
    train_df = X_train.copy()
    train_df[target] = y_train.values

    test_df = X_test.copy()
    test_df[target] = y_test.values

    # simpan output
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # plot distribusi
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(df["jumlah_pesanan"], bins=20)
    plt.title("Distribusi Jumlah Pesanan")
    plt.xlabel("Jumlah Pesanan")
    plt.ylabel("Frekuensi")

    plt.subplot(1, 2, 2)
    plt.hist(df["harga_satuan"], bins=15)
    plt.title("Distribusi Harga Satuan")
    plt.xlabel("Harga Satuan")
    plt.ylabel("Frekuensi")

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    # simpan ringkasan data
    summary = {
        "rows_total": int(df.shape[0]),
        "cols_total": int(df.shape[1]),
        "rows_train": int(train_df.shape[0]),
        "rows_test": int(test_df.shape[0]),
        "fitur": fitur,
        "target": target,
        "missing_value_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Prepare selesai.")
    print(f"Train saved to: {TRAIN_PATH}")
    print(f"Test saved to : {TEST_PATH}")
    print(f"Plot saved to : {PLOT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()