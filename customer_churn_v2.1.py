"""
Customer Churn Analysis (Modernized + Optional Improvements) — v2
Fixes:
  • Corrected precision/recall extraction for positive class
  • Casted dtypes for plotting to avoid matplotlib categorical-unit messages
  • Added clearer guidance for DESIRED_FEATURES case-sensitivity
  • Normalized column case for Gender and Tenure
"""

from __future__ import annotations

# === Imports ===
from pathlib import Path
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.units").setLevel(logging.ERROR)
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress "Using categorical units..." chatter
#warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)

# === Config ===
sns.style="whitegrid"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATA_URL = "https://raw.githubusercontent.com/PAWinTX/Telco_Customer_Churn_with_Python/master/local_WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path("C:/Github/Telco")  # change as needed
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"

# IMPORTANT: Column names are case-sensitive and this dataset uses lower-case for several,
# e.g., 'gender', 'tenure'. If you set DESIRED_FEATURES, use the exact column names from the CSV.
DESIRED_FEATURES: Optional[List[str]] = None
# Example:
# DESIRED_FEATURES = [
#     "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
#     "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
#     "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
#     "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
#     "MonthlyCharges", "TotalCharges"
# ]


def load_data() -> pd.DataFrame:
    cd = pd.read_csv(DATA_URL)
    logging.info("Loaded data: %s rows, %s columns", *cd.shape)
    return cd


def clean_data(cd: pd.DataFrame) -> pd.DataFrame:
    before = len(cd)
    cd = cd.drop_duplicates(keep="first")
    logging.info("Dropped %d duplicate rows", before - len(cd))

    # Normalize column names for consistency
    if "gender" in cd.columns:
        cd.rename(columns={"gender": "Gender"}, inplace=True)
    if "tenure" in cd.columns:
        cd.rename(columns={"tenure": "Tenure"}, inplace=True)

    # Coerce TotalCharges to numeric (handles spaces or junk), then impute 0.0
    if "TotalCharges" in cd.columns:
        cd["TotalCharges"] = pd.to_numeric(cd["TotalCharges"], errors="coerce").fillna(0.0)

    # Map common binary categories to 0/1 integers
    bin_maps = {
        "gender": {"Male": 1, "Female": 0},
        "Gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "Churn": {"Yes": 1, "No": 0},
    }
    for col, mapping in bin_maps.items():
        if col in cd.columns:
            cd[col] = cd[col].map(mapping).astype("Int64")

    # Optional: compute sanity-check tenure using charges
    if "MonthlyCharges" in cd.columns and "TotalCharges" in cd.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            tenure_calc = np.where(cd["MonthlyCharges"] > 0,
                                   np.round(cd["TotalCharges"] / cd["MonthlyCharges"]),
                                   np.nan)
        cd["TenureCalc"] = pd.Series(tenure_calc).astype("Int64")

    # Drop ID columns if present (non-predictive)
    for id_col in ["customerID", "CustomerID", "CustomerId"]:
        if id_col in cd.columns:
            cd = cd.drop(columns=[id_col])

    return cd


# === Plot Helpers ===
def plot_count_by_churn(df: pd.DataFrame, col: str, save_dir: Path) -> None:
    if TARGET not in df.columns or col not in df.columns:
        return
    x = df[col]
    h = df[TARGET].astype("Int64").astype(str)
    plt.figure()
    sns.countplot(x=x, hue=h)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Count of {col} by {TARGET}")
    plt.tight_layout()
    fig_path = save_dir / f"count_{col}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", fig_path)


def plot_violin_numeric(df: pd.DataFrame, col: str, save_dir: Path) -> None:
    if TARGET not in df.columns or col not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[col]):
        return
    y = pd.to_numeric(df[col], errors="coerce").astype(float)
    x = df[TARGET].astype("Int64").astype(int)
    plt.figure()
    sns.violinplot(x=x, y=y)
    plt.xlabel(TARGET)
    plt.ylabel(col)
    plt.title(f"Distribution of {col} by {TARGET}")
    plt.tight_layout()
    fig_path = save_dir / f"violin_{col}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", fig_path)


def quick_eda(cd: pd.DataFrame) -> None:
    logging.info("Numeric summary:\n%s", cd.describe().to_string())

    for col in ["Tenure", "TenureCalc", "MonthlyCharges", "TotalCharges"]:
        if col in cd.columns and pd.api.types.is_numeric_dtype(cd[col]):
            vals = pd.to_numeric(cd[col], errors="coerce")
            plt.figure()
            plt.hist(vals.dropna())
            plt.xlabel(col)
            plt.ylabel("Customers")
            plt.title(f"Histogram of {col}")
            plt.tight_layout()
            fig_path = FIG_DIR / f"hist_{col}.png"
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close()
            logging.info("Saved %s", fig_path)

    cat_candidates = cd.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_candidates[:10]:
        plot_count_by_churn(cd, c, FIG_DIR)

    num_candidates = cd.select_dtypes(include=["number"]).columns.tolist()
    for c in [c for c in num_candidates if c != TARGET][:8]:
        plot_violin_numeric(cd, c, FIG_DIR)


def select_features(cd: pd.DataFrame, target: str, desired: Optional[List[str]] = None) -> pd.DataFrame:
    if target not in cd.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")
    if desired is None:
        feats = [c for c in cd.columns if c != target]
    else:
        feats = [c for c in desired if c in cd.columns]
        missing = set(desired) - set(feats)
        if missing:
            logging.warning("Desired features missing and will be ignored: %s", sorted(missing))
        if not feats:
            raise ValueError("No desired features found in dataframe.")
    return cd[feats + [target]].copy()


def build_model(cd: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    cd2 = select_features(cd, TARGET, DESIRED_FEATURES)

    X = cd2.drop(columns=[TARGET])
    y = cd2[TARGET].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=400,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    acc = pipe.score(X_test, y_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )
    precision_pos = float(prec_arr[1])
    recall_pos = float(rec_arr[1])
    f1_pos = float(f1_arr[1])

    logging.info("Accuracy: %.3f | ROC AUC: %.3f", acc, auc)
    logging.info("Confusion matrix (labels=[0,1]):\n%s", cm)
    logging.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logging.info(
        "Positive class — precision: %.3f, recall: %.3f, F1: %.3f",
        precision_pos,
        recall_pos,
        f1_pos,
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_acc = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy").mean()
    logging.info("10-fold CV accuracy: %.3f", cv_acc)

    return pipe, X_test, y_test


def export_outputs(cd: pd.DataFrame) -> None:
    csv_path = OUTPUT_DIR / "cleaned_WA_Fn-UseC_-Telco-Customer-Churn.csv"
    cd.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info("Wrote %s", csv_path)

    xlsx_path = OUTPUT_DIR / "cleaned_WA_Fn-UseC_-Telco-Customer-Churn.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        cd.to_excel(writer, sheet_name="CustData", index=False)
    logging.info("Wrote %s", xlsx_path)


def main() -> None:
    cd = load_data()
    cd = clean_data(cd)
    quick_eda(cd)
    _pipe, _X_test, _y_test = build_model(cd)
    export_outputs(cd)


if __name__ == "__main__":
    main()
