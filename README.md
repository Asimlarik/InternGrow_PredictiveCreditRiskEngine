# InternGrow_PredictiveCreditRiskEngine

**InternGrow Machine Learning Track — Task 1**

## 📌 Objective
Evaluate creditworthiness using classification models on financial history data.

## 🧠 Approach
- Logistic Regression and Decision Tree models, plus Random Forest, evaluated via F1-Score and ROC-AUC
- **Upgrade Feature**: **SMOTE** (Synthetic Minority Oversampling Technique) applied to the training set to handle the realistic class imbalance (~12% default rate), plus a detailed **feature importance** plot

## 📂 Files
- `predictive_credit_risk_engine.py` — full pipeline
- `roc_curves.png`, `feature_importance.png` — generated after running

## ▶️ How to Run
```bash
pip install -r requirements.txt
python predictive_credit_risk_engine.py
```

## 📈 Results (example run)
| Model | F1-Score | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.494 | **0.880** |
| Decision Tree | 0.378 | 0.768 |
| Random Forest | 0.462 | 0.840 |

SMOTE meaningfully improved recall on the minority (default) class compared to training on the raw imbalanced data — critical for a credit risk model, where missing an actual default is costlier than a false alarm.

## 🎥 Submission Checklist (InternGrow)
- [ ] Push to GitHub as `InternGrow_PredictiveCreditRiskEngine`
- [ ] Record project video, post on LinkedIn tagging @InternGrow, with GitHub link
- [ ] Submit via the InternGrow submission form
