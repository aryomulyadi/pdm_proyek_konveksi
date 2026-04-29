import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression



# PATH


TRAIN_PATH = "data/prepared/train.csv"
TEST_PATH = "data/prepared/test.csv"

MODEL_DIR = "models"
METRICS_DIR = "metrics"

EXPERIMENT_LOG_PATH = os.path.join(METRICS_DIR, "experiment_log.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_MODEL_INFO_PATH = os.path.join(METRICS_DIR, "best_model.json")
BEST_MODEL_PREDICTIONS_PATH = os.path.join(METRICS_DIR, "best_model_predictions.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# LOAD DATA


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# FITUR DAN TARGET

target = "harga_satuan"

features = [
    "jumlah_pesanan",
    "bahan",
    "model_lengan",
    "sablon_depan",
    "sablon_belakang",
    "sablon_lengan_kiri",
    "sablon_lengan_kanan",
    "sablon_bawah"
]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# PREPROCESSING

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


def create_preprocessor():
    """
    Membuat preprocessor baru untuk setiap model.
    Ini penting agar setiap pipeline berdiri sendiri.
    """

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", encoder, categorical_features)
        ]
    )

    return preprocessor


# DAFTAR MODEL


models = {
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=8,
        min_samples_leaf=5
    ),

    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    ),

    "Linear Regression": LinearRegression()
}



# FUNGSI EVALUASI

def calculate_metrics(y_true, y_pred):
    """
    Menghitung MAE, RMSE, dan R2.
    RMSE dihitung manual agar kompatibel dengan berbagai versi scikit-learn.
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# TRAINING DAN EVALUASI MODEL

experiment_results = []
trained_pipelines = {}

print("\nMEMULAI PROSES MODELING")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nMelatih model: {model_name}")

    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Training model
    pipeline.fit(X_train, y_train)

    # Prediksi train dan test
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    # Hitung metrik train
    train_mae, train_rmse, train_r2 = calculate_metrics(y_train, train_pred)

    # Hitung metrik test
    test_mae, test_rmse, test_r2 = calculate_metrics(y_test, test_pred)

    # Simpan hasil eksperimen
    result = {
        "model": model_name,
        "train_mae": round(train_mae, 2),
        "train_rmse": round(train_rmse, 2),
        "train_r2": round(train_r2, 4),
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4)
    }

    experiment_results.append(result)
    trained_pipelines[model_name] = pipeline

    # Simpan model masing-masing
    model_file_name = model_name.lower().replace(" ", "_") + ".pkl"
    model_path = os.path.join(MODEL_DIR, model_file_name)
    joblib.dump(pipeline, model_path)

    print("Hasil evaluasi:")
    print(f"Train MAE  : {result['train_mae']}")
    print(f"Train RMSE : {result['train_rmse']}")
    print(f"Train R2   : {result['train_r2']}")
    print(f"Test MAE   : {result['test_mae']}")
    print(f"Test RMSE  : {result['test_rmse']}")
    print(f"Test R2    : {result['test_r2']}")
    print("-" * 70)


# SIMPAN LOG EKSPERIMEN

experiment_df = pd.DataFrame(experiment_results)

experiment_df = experiment_df.sort_values(
    by=["test_mae", "test_rmse", "test_r2"],
    ascending=[True, True, False]
)

experiment_df.to_csv(EXPERIMENT_LOG_PATH, index=False)


# PILIH MODEL TERBAIK

best_result = experiment_df.iloc[0].to_dict()
best_model_name = best_result["model"]
best_pipeline = trained_pipelines[best_model_name]

joblib.dump(best_pipeline, BEST_MODEL_PATH)

with open(BEST_MODEL_INFO_PATH, "w") as file:
    json.dump(best_result, file, indent=4)


# SIMPAN PREDIKSI MODEL TERBAIK


best_predictions = best_pipeline.predict(X_test)

prediction_df = test_df.copy()
prediction_df["prediksi_harga_satuan"] = best_predictions.round(2)
prediction_df["selisih_error"] = (
    prediction_df["harga_satuan"] - prediction_df["prediksi_harga_satuan"]
).abs().round(2)

prediction_df.to_csv(BEST_MODEL_PREDICTIONS_PATH, index=False)


# OUTPUT AKHIR

print("\nLOG EKSPERIMEN TERSIMPAN DI:")
print(EXPERIMENT_LOG_PATH)

print("\nMODEL TERBAIK")
print("=" * 70)
print(f"Nama model : {best_model_name}")
print(f"Test MAE   : {best_result['test_mae']}")
print(f"Test RMSE  : {best_result['test_rmse']}")
print(f"Test R2    : {best_result['test_r2']}")
print("=" * 70)

print("\nFILE YANG DIHASILKAN:")
print(f"1. {EXPERIMENT_LOG_PATH}")
print(f"2. {BEST_MODEL_INFO_PATH}")
print(f"3. {BEST_MODEL_PATH}")
print(f"4. {BEST_MODEL_PREDICTIONS_PATH}")