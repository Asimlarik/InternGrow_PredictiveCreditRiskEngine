"""
InternGrow Machine Learning Track - Task 1
Predictive Credit Risk Engine
------------------------------------------------
Evaluates creditworthiness using classification models, with an
UPGRADE FEATURE: SMOTE (Synthetic Minority Oversampling Technique) to
handle imbalanced financial datasets, plus detailed feature importance
plots.

Author: (your name here)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE
from collections import Counter

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------
# 1. DATA GENERATION (imbalanced by design - ~12% default rate)
# --------------------------------------------------------------------------
def load_data(csv_path: str = None, n_samples: int = 3000) -> pd.DataFrame:
    """
    Generates a realistic, DELIBERATELY IMBALANCED synthetic credit
    dataset (~12% bad-credit rate, mirroring real-world financial data
    where defaults are the minority class). Pass csv_path to use a real
    dataset instead (e.g. UCI German Credit / Kaggle credit risk data)
    with a binary 'default' target column.
    """
    if csv_path:
        return pd.read_csv(csv_path)

    age = np.random.randint(21, 65, n_samples)
    income = np.random.normal(55000, 20000, n_samples).clip(12000, 200000)
    debt = np.random.normal(15000, 10000, n_samples).clip(0, 150000)
    credit_lines = np.random.randint(0, 15, n_samples)
    loan_amount = np.random.normal(20000, 15000, n_samples).clip(500, 150000)
    late_payments = np.random.poisson(1.2, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    credit_history_length = np.random.randint(0, 30, n_samples)
    debt_to_income = debt / income

    risk_score = (
        0.000009 * debt - 0.00002 * income + 0.35 * late_payments
        + 0.9 * debt_to_income - 0.05 * employment_years
        - 0.03 * credit_history_length + 0.06 * credit_lines
        + np.random.normal(0, 1.0, n_samples)
    )
    threshold = np.percentile(risk_score, 88)  # ~12% default rate -> imbalanced
    default = (risk_score > threshold).astype(int)

    return pd.DataFrame({
        "age": age, "income": income.round(2), "debt": debt.round(2),
        "credit_lines": credit_lines, "loan_amount": loan_amount.round(2),
        "late_payments": late_payments, "employment_years": employment_years,
        "credit_history_length": credit_history_length,
        "debt_to_income": debt_to_income.round(4),
        "loan_to_income": (loan_amount / income).round(4),
        "default": default,
    })


# --------------------------------------------------------------------------
# 2. TRAIN / EVAL
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return {"name": name, "f1": f1, "roc_auc": auc, "y_proba": y_proba}


def plot_feature_importance(model, feature_names, out_path="feature_importance.png"):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx], color="darkorange")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance")
    plt.title("Feature Importance - Random Forest (Credit Risk)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Feature importance plot saved to {out_path}")


def plot_roc_curves(results, y_test, out_path="roc_curves.png"):
    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Credit Risk Models (after SMOTE)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"ROC curve plot saved to {out_path}")


def main():
    df = load_data()
    X = df.drop(columns=["default"])
    y = df["default"]

    print(f"Original class distribution: {Counter(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---- UPGRADE FEATURE: SMOTE oversampling on the training set only ----
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE (train set only): {Counter(y_train_res)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    rf_ref = None
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train_res)
            res = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train_res, y_train_res)
            res = evaluate_model(name, model, X_test, y_test)
            if name == "Random Forest":
                rf_ref = model
        results.append(res)

    plot_roc_curves(results, y_test)
    if rf_ref is not None:
        plot_feature_importance(rf_ref, X.columns.tolist())

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best['name']} ({best['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
