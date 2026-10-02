from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRICE_CANDIDATES = ["Price","Used/New Price","Used Price","Selling Price","Sale Price","Market Price","MSRP"]

def choose_target(df):
    for c in PRICE_CANDIDATES:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 20 and (s > 0).sum() >= 20:
                return c
    raise ValueError("No usable price target found. Upload used-market data containing a price column.")

def train(df, artifact_dir="artifacts"):
    d = Path(artifact_dir); d.mkdir(exist_ok=True)
    target = choose_target(df)
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna() & (y > 0)
    df = df.loc[valid].copy(); y = y.loc[valid]

    features = [c for c in df.columns if c != target]
    # Don't leak target-like fields.
    leak = {"Price","Used/New Price","Used Price","Selling Price","Sale Price","Market Price"}
    features = [c for c in features if c not in leak]
    X = df[features].copy()
    features = [c for c in features if X[c].notna().any()]
    X = X[features]

    # Convert date into useful numeric recency fields.
    for c in list(X.columns):
        if "date" in c.lower() or c == "Listing Date":
            dt = pd.to_datetime(X[c], errors="coerce")
            X[c + " Year"] = dt.dt.year
            X[c + " Month"] = dt.dt.month
            X = X.drop(columns=[c])
            features = [x for x in features if x != c] + [c + " Year", c + " Month"]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    models = {
        "Ridge": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2),
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42)
    results, fitted = {}, {}
    for name, est in models.items():
        pipe = Pipeline([("preprocessor", prep), ("model", est)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {
            "MAE": float(mean_absolute_error(yte, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "R2": float(r2_score(yte, pred))
        }
        fitted[name] = pipe

    best_name = min(results, key=lambda k: results[k]["MAE"])
    joblib.dump(fitted[best_name], d / "best_model.joblib")

    meta = {
        "target": target,
        "features": list(X.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "best_model": best_name,
        "metrics": results[best_name],
        "all_metrics": results,
        "training_rows": int(len(X)),
        "price_min": float(y.min()),
        "price_max": float(y.max()),
        "used_market_model": "Mileage, condition, location, accident history and listing date are supported when supplied."
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta

def load(artifact_dir="artifacts"):
    d = Path(artifact_dir)
    return joblib.load(d / "best_model.joblib"), json.loads((d / "metadata.json").read_text())
