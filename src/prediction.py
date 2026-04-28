import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, mean_absolute_error

def predict_model():
    df = pd.read_csv("data/tmdb_movies.csv")

    df = df[df["budget"] > 10000]
    df = df[df["revenue"] > 10000]
    df = df[df["runtime"] > 30]

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])

    df["budget_log"] = np.log1p(df["budget"])

    df["roi"] = ((df["revenue"] - df["budget"]) / df["budget"]).clip(lower=-0.99, upper=10)
    df["success"] = (df["roi"] > 1.5).astype(int)
    df["roi_log"] = np.log1p(df["roi"])

    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month

    def extract_genres(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    df["genre_names"] = df["genres"].apply(extract_genres)

    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(
        mlb.fit_transform(df["genre_names"]), columns=mlb.classes_, index=df.index
    )

    def extract_companies(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
        
    df["company_names"] = df["production_companies"].apply(extract_companies)

    all_companies = pd.Series([c for sub in df["company_names"] for c in sub])
    top_companies = all_companies.value_counts().head(100).index

    df["company_filtered"] = df["company_names"].apply(
        lambda lst: [c for c in lst if c in top_companies]
    )

    mlb_comp = MultiLabelBinarizer()
    comp_df = pd.DataFrame(
        mlb_comp.fit_transform(df["company_filtered"]),
        columns=["comp_" + c for c in mlb_comp.classes_],
        index=df.index,
    )

    X = pd.concat(
        [
            df[["budget_log", "runtime", "year", "month"]],
            genre_df,
            comp_df,
        ],
        axis=1,
    )

    y_clf = df["success"]
    y_reg = df["roi_log"]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_clf, y_reg, test_size=0.2, random_state=42)
    )

    clf_model = RandomForestClassifier(n_estimators=200, min_samples_split= 20, random_state=42, class_weight="balanced", n_jobs=-1)
    clf_model.fit(X_train, y_clf_train)

    reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_model.fit(X_train, y_reg_train)

    clf_preds = clf_model.predict(X_test)
    reg_preds = reg_model.predict(X_test)

    print(
        "Classification Accuracy with over view:", accuracy_score(y_clf_test, clf_preds)
    )
    roi_preds = np.expm1(reg_preds)
    roi_true = np.expm1(y_reg_test)

    print("ROI MAE with over view:", mean_absolute_error(roi_true, roi_preds))

    baseline = y_reg_train.mean()
    baseline_preds = [baseline] * len(y_reg_test)

    baseline_roi_preds = np.expm1(baseline_preds)
    print("Baseline ROI MAE:", mean_absolute_error(roi_true, baseline_roi_preds))

    improvement = (mean_absolute_error(roi_true, baseline_roi_preds) - mean_absolute_error(roi_true, roi_preds)) / mean_absolute_error(roi_true, baseline_roi_preds)
    
    print(f"Improvement over baseline: {improvement:.2%}")

    new_movie = {
        "budget_log": np.log1p(150000000),
        "runtime": 100,
        "year": 2019,
        "month": 8,

        "Action": 1,
        "Adventure": 1,
        "Science Fiction": 1,

        "comp_Warner Bros. Pictures": 1,
        "comp_Legendary Pictures": 1,
    }

    row = {col: 0 for col in X.columns}
    row.update(new_movie)

    X_new = pd.DataFrame([row])

    reg_pred = reg_model.predict(X_new)[0]

    roi = np.expm1(reg_pred)

    clf_pred = clf_model.predict(X_new)[0]
    clf_prob = clf_model.predict_proba(X_new)[0][1]

    label = "Hit" if roi > 1.5 else "Flop"

    print("\n--- New Movie Prediction ---")
    print(f"Predicted ROI: {roi:.2f}")
    print(f"Result (based on ROI): {label}")
    print(f"Classifier probability: {clf_prob:.2f}")
    
    
    return clf_model, reg_model, X_train.columns


if __name__ == "__main__":
    clf_model, reg_model, feature_columns = predict_model()
