import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from alibi_detect.cd import KSDrift
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from sklearn.preprocessing import StandardScaler

from prometheus_client import start_http_server, Counter, Gauge



logger = get_logger(__name__)

app = Flask(__name__)  # templates/ and static/ are defaults

prediction_counter = Counter('prediction_count', 'Number of predictions made')
drift_count = Counter('drift_count', 'Number of times drift was detected')


MODEL_PATH = "artifacts/models/random_forest_model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = [
    "Age", "Pclass", "Fare", "Sex", "Embarked",
    "FamilySize", "HasCabin", "Isalone", "Title",
    "Pclass_Fare", "Age_Fare"
]

feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_training_data() -> np.ndarray:
    """
    Fit scaler on training data from feature store using ONLY model features,
    in a stable column order matching FEATURE_NAMES.
    Returns scaled training matrix (numpy) for KSDrift reference.
    """
    # Retrieve training features from Redis
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)
    all_features_df = pd.DataFrame.from_dict(all_features, orient="index")

    # Keep exactly the model inputs; drop rows with any missing values for scaler fitting
    missing_cols = [c for c in FEATURE_NAMES if c not in all_features_df.columns]
    if missing_cols:
        raise ValueError(f"Feature store is missing expected columns: {missing_cols}")

    X_train = all_features_df[FEATURE_NAMES].copy()

    # If necessary, coerce dtypes
    for col in FEATURE_NAMES:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_train = X_train.dropna(axis=0, how="any")

    # Fit scaler ONLY on model inputs (not on target), then return scaled matrix for drift
    scaler.fit(X_train.to_numpy())
    return scaler.transform(X_train.to_numpy())

# Fit scaler and initialize KSDrift with the scaled historical data
historical_data = fit_scaler_on_training_data()  # np.ndarray
ksd = KSDrift(historical_data, p_val=0.05)

TITLE_MAP = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}  # else → Rare (4)

def map_title(name: str) -> int:
    if not name:
        return 4
    m = re.search(r" ([A-Za-z]+)\.", name)
    t = m.group(1) if m else None
    return TITLE_MAP.get(t, 4)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form

        # ---- Raw inputs (with safe defaults) ----
        Name    = f.get("Name", "").strip()
        Age     = float(f.get("Age", 28) or 28)
        Pclass  = int(f.get("Pclass", 3) or 3)
        Fare    = float(f.get("Fare", 7.25) or 7.25)

        # Encoded per your mapping
        Sex       = int(f.get("Sex", 0) or 0)        # male=0, female=1
        Embarked  = int(f.get("Embarked", 0) or 0)  # S=0, C=1, Q=2

        # For derived features
        SibSp   = int(f.get("SibSp", 0) or 0)
        Parch   = int(f.get("Parch", 0) or 0)
        Cabin   = f.get("Cabin", "").strip()

        # ---- Derived server-side features ----
        FamilySize  = SibSp + Parch + 1
        HasCabin    = 1 if Cabin else 0
        Isalone     = 1 if FamilySize == 1 else 0
        Title       = map_title(Name)
        Pclass_Fare = Pclass * Fare
        Age_Fare    = Age * Fare

        # Build X in the exact order the model expects
        X_df = pd.DataFrame([[
            Age, Pclass, Fare, Sex, Embarked,
            FamilySize, HasCabin, Isalone, Title,
            Pclass_Fare, Age_Fare
        ]], columns=FEATURE_NAMES)

        # ----- Drift detection (scale first using SAME columns/order) -----
        X_scaled = scaler.transform(X_df.to_numpy())  # use numpy to avoid feature-name checks
        drift = ksd.predict(X_scaled.astype(np.float32))
        print("Drift response: %s", drift)

        drift_response = drift.get('data', {})
        is_drift = drift_response.get('is_drift', None)

        if is_drift is not None and is_drift==1:
            print("Drift detected for input data.")
            logger.warning("Data drift detected for input data.")
            
            drift_count.inc()
            # You can choose to handle drift here (e.g., reject prediction, log, etc.)

        # ---- Model inference ----
        pred = int(model.predict(X_df)[0])  # if your model was fit with pandas, this is fine
        prediction_counter.inc()

        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_df)[0]
            # robustly pick class 1
            if hasattr(model, "classes_"):
                class_idx = int(np.where(model.classes_ == 1)[0][0])
            else:
                class_idx = 1
            prob = float(proba[class_idx])

        result_text = "Survived" if pred == 1 else "Did not survive"

        features_raw = dict(
            Name=Name, Age=Age, Pclass=Pclass, Fare=Fare,
            Sex=Sex, Embarked=Embarked, SibSp=SibSp, Parch=Parch, Cabin=Cabin
        )
        features_computed = dict(
            FamilySize=FamilySize, HasCabin=HasCabin, Isalone=Isalone,
            Title=Title, Pclass_Fare=Pclass_Fare, Age_Fare=Age_Fare
        )

        return render_template(
            "index.html",
            prediction=result_text,
            prediction_prob=prob,
            features=features_raw,
            features_computed=features_computed
        )

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 400
@app.route("/metrics")
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest(), content_type='text/plain; version=0.0.4; charset=utf-8')

if __name__ == "__main__":
    start_http_server(8000)  # Expose metrics at :8000
    app.run(debug=True, host='0.0.0.0', port=5000)
