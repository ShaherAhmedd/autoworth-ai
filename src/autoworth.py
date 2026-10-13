"""Data loading and model utilities for the AutoWorth Streamlit app."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


DATA_FILES = {
    "audi.csv": "Audi",
    "bmw.csv": "BMW",
    "ford.csv": "Ford",
    "hyundi.csv": "Hyundai",
    "merc.csv": "Mercedes",
    "skoda.csv": "Skoda",
    "toyota.csv": "Toyota",
    "vauxhall.csv": "Vauxhall",
    "vw.csv": "Volkswagen",
}
REQUIRED_COLUMNS = {
    "model", "year", "price", "transmission", "mileage", "fuelType",
    "tax", "mpg", "engineSize",
}
NUMERIC_FEATURES = [
    "year", "mileage", "tax", "mpg", "engineSize", "CarAge",
    "MileagePerYear", "CarAgeSquared",
]
CATEGORICAL_FEATURES = ["Make", "model", "transmission", "fuelType"]
REFERENCE_YEAR = 2021  # Matches the notebook's feature engineering.


def _read_csv_bytes(name: str, content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content))
    frame.columns = [str(column).strip() for column in frame.columns]
    if "tax(£)" in frame.columns and "tax" not in frame.columns:
        frame = frame.rename(columns={"tax(£)": "tax"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    frame["Make"] = DATA_FILES[name.lower()]
    for col in ("model", "transmission", "fuelType", "Make"):
        frame[col] = frame[col].astype("string").str.strip()
    for col in ("year", "price", "mileage", "tax", "mpg", "engineSize"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_dataset_files(files: dict[str, bytes]) -> pd.DataFrame:
    """Load the nine expected manufacturer CSV files from a filename/bytes map."""
    normalized = {Path(name).name.lower(): content for name, content in files.items()}
    missing = sorted(set(DATA_FILES).difference(normalized))
    if missing:
        raise ValueError("Missing required CSV files: " + ", ".join(missing))
    frames = [_read_csv_bytes(name, normalized[name]) for name in DATA_FILES]
    return clean_dataset(pd.concat(frames, ignore_index=True))


def load_zip(content: bytes) -> pd.DataFrame:
    """Read expected CSVs from a ZIP archive, ignoring unrelated entries."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = {
                Path(item.filename).name.lower(): archive.read(item)
                for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name.lower() in DATA_FILES
            }
    except BadZipFile as exc:
        raise ValueError("That file is not a valid ZIP archive.") from exc
    return load_dataset_files(entries)


def load_local_dataset(data_dir: Path) -> pd.DataFrame | None:
    files = {}
    for name in DATA_FILES:
        path = data_dir / name
        if path.exists():
            files[name] = path.read_bytes()
    if not files:
        return None
    return load_dataset_files(files)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's duplicate and impossible-year cleanup."""
    frame = frame.copy().drop_duplicates()
    frame = frame[frame["year"].le(REFERENCE_YEAR) & frame["price"].gt(0)]
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No usable records remain after cleaning the CSV data.")
    return frame


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the same age and usage features used in the notebook."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        result["CarAge"] = REFERENCE_YEAR - result["year"]
        result["MileagePerYear"] = result["mileage"] / result["CarAge"].replace(0, 1)
        result["CarAgeSquared"] = result["CarAge"] ** 2
        return result


def build_model() -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessing", prep),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_model(frame: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    features = frame.drop(columns="price")
    target = frame["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.15, random_state=42
    )
    # Keep the notebook's 70/15/15 split by reserving validation rows first.
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=(15 / 85), random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, predicted)),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "test_rows": float(len(y_test)),
    }
    return model, metrics


def make_prediction_row(
    *, make: str, model: str, year: int, mileage: int, transmission: str,
    fuel_type: str, tax: float, mpg: float, engine_size: float,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "Make": make,
        "model": model,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
    }])
