import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, mean_absolute_error


def predict():
    df = pd.read_csv("data/tmdb_movies.csv")

    df = df[df["budget"] > 1000]
    df = df[df["revenue"] > 1000]
    df = df[df["runtime"] > 40]

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])

    df["roi"] = df["revenue"] / df["budget"]
    df["roi"] = df["roi"].clip(upper=10)

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

    df = pd.concat([df, genre_df], axis=1)

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

    df = pd.concat([df, comp_df], axis=1)

    features = [
        "budget",
        "runtime",
        # "popularity",
        # "vote_average",
        # "vote_count",
        "year",
        "month",
    ] + list(genre_df.columns) + list(comp_df.columns)

    X = df[features]

    y_clf = df["success"]
    y_reg = df["roi_log"]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_clf, y_reg, test_size=0.2, random_state=42)
    )
    clf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    clf_model.fit(X_train, y_clf_train)

    reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
    reg_model.fit(X_train, y_reg_train)

    clf_preds = clf_model.predict(X_test)
    reg_preds = reg_model.predict(X_test)

    print("Classification Accuracy:", accuracy_score(y_clf_test, clf_preds))
    print("ROI MAE:", mean_absolute_error(y_reg_test, reg_preds))

if __name__ == "__main__":
    predict()
