"""
Phase 5 of the BRCA1/2 reproduction pipeline: ensembles on the same
internal train/test split as Phase 4 (chat_history.md SS19).

Soft Voting (RF+GB+XGB) and Stacking (RF+GB+XGB base, logistic
regression meta) -- same architecture as the original methodology
(chat_history.md SS3). Reuses the already-tuned best estimators from
Phase 4 as base learners rather than re-tuning from scratch.

2026-07-25 (Stage 4): added BayesDel_noAF_score as a 12th score -- see
new_scripts/04_filter_and_report.py's docstring for the decision rationale.
"""
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

SCORE_COLS = [
    "ClinPred_score", "MetaRNN_score", "REVEL_score", "MetaSVM_score", "MetaLR_score",
    "VEST4_score", "AlphaMissense_score", "PrimateAI_score", "CADD_raw", "CADD_phred",
    "ESM1b_score", "BayesDel_noAF_score", "Eigen_raw_coding",
]


def evaluate(name, model, X_test, y_test, results):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    results[name] = {
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "test_f1": round(f1_score(y_test, pred, zero_division=0), 4),
    }
    print(name, results[name])


def main():
    train = pd.read_csv("pipeline/internal_train_split.csv")
    test = pd.read_csv("pipeline/internal_test_split.csv")
    X_train, y_train = train[SCORE_COLS], train["label"].astype(int)
    X_test, y_test = test[SCORE_COLS], test["label"].astype(int)

    rf = joblib.load("pipeline/models/model_randomforest.pkl")
    gb = joblib.load("pipeline/models/model_gradientboosting.pkl")
    xgb = joblib.load("pipeline/models/model_xgboost.pkl")
    svm = joblib.load("pipeline/models/model_svm.pkl")

    results = {}

    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("xgb", xgb)], voting="soft"
    )
    voting.fit(X_train, y_train)
    evaluate("ML_VotingEnsemble", voting, X_test, y_test, results)
    joblib.dump(voting, "pipeline/models/model_voting.pkl")

    stacking = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("xgb", xgb), ("svm", svm)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=5,
    )
    stacking.fit(X_train, y_train)
    evaluate("StackingEnsemble", stacking, X_test, y_test, results)
    joblib.dump(stacking, "pipeline/models/model_stacking.pkl")

    prior = pd.read_csv("pipeline/table_internal_comparison.csv", index_col=0)
    combined = pd.concat([prior[["test_roc_auc", "test_accuracy", "test_precision",
                                  "test_recall", "test_f1"]], pd.DataFrame(results).T])
    combined.to_csv("pipeline/table_internal_comparison.csv")
    print("\n=== Full internal comparison (base models + ensembles) ===")
    print(combined)


if __name__ == "__main__":
    main()
