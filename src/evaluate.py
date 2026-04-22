import os
import json
import pickle
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TEST_PATH = "data/prepared/test.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics/eval_metrics.json"
PREDICTIONS_PATH = "metrics/test_predictions.csv"


def main():
    os.makedirs("metrics", exist_ok=True)

    # baca data test
    df = pd.read_csv(TEST_PATH)

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

    X_test = df[fitur]
    y_test = df[target]

    # load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # prediksi
    y_pred = model.predict(X_test)

    # metrik evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_test": int(len(df)),
        "model": "DecisionTreeRegressor"
    }

    # simpan metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # simpan hasil prediksi
    hasil_prediksi = X_test.copy()
    hasil_prediksi["harga_aktual"] = y_test.values
    hasil_prediksi["harga_prediksi"] = y_pred
    hasil_prediksi["selisih"] = hasil_prediksi["harga_aktual"] - hasil_prediksi["harga_prediksi"]

    hasil_prediksi.to_csv(PREDICTIONS_PATH, index=False)

    print("Evaluasi selesai.")
    print(f"Metrics saved to     : {METRICS_PATH}")
    print(f"Predictions saved to : {PREDICTIONS_PATH}")
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()