import os
import json
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_PATH = "data/prepared/train.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics/train_metrics.json"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # baca data train
    df = pd.read_csv(TRAIN_PATH)

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

    X = df[fitur]
    y = df[target]

    # pisahkan fitur numerik dan kategorikal
    numeric_features = ["jumlah_pesanan"]
    categorical_features = [
        "bahan",
        "model_lengan",
        "sablon_depan",
        "sablon_belakang",
        "sablon_lengan_kiri",
        "sablon_lengan_kanan",
        "sablon_bawah"
    ]

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    # pipeline model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])

    # training
    model.fit(X, y)

    # prediksi pada data train
    y_pred = model.predict(X)

    # metrik training
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": int(len(df)),
        "model": "DecisionTreeRegressor"
    }

    # simpan model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # simpan metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("Training selesai.")
    print(f"Model saved to   : {MODEL_PATH}")
    print(f"Metrics saved to : {METRICS_PATH}")
    print("Train metrics:")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()