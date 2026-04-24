import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv("data/tmdb_movies.csv")

df = df[df["budget"] > 1000]
df = df[df["revenue"] > 1000]
df = df[df["runtime"] > 40]

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])

df["roi"] = (df["revenue"] / df["budget"]).clip(upper=10)
df["success"] = (df["roi"] > 2).astype(int)
df["roi_log"] = np.log1p(df["roi"])

df["year"] = df["release_date"].dt.year
df["month"] = df["release_date"].dt.month

def extract_genres(x):
    try:
        items = ast.literal_eval(x)
        return [g["name"] for g in items if isinstance(g, dict)]
    except:
        return []

df["genre_names"] = df["genres"].apply(extract_genres)

mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(
    mlb.fit_transform(df["genre_names"]),
    columns=mlb.classes_,
    index=df.index
)

def extract_companies(x):
    try:
        items = ast.literal_eval(x)
        return [c["name"] for c in items]
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
    index=df.index
)

X = pd.concat([
    df[["budget","runtime","popularity","vote_average","vote_count","year","month"]],
    genre_df,
    comp_df
], axis=1)

y_clf = df["success"]
y_reg = df["roi_log"]

X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)

clf_model = RandomForestClassifier(n_estimators=150, random_state=42)
clf_model.fit(X_train, y_clf_train)

reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
reg_model.fit(X_train, y_reg_train)

clf_preds = clf_model.predict(X_test)
reg_preds = reg_model.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_clf_test, clf_preds))
print("ROI MAE:", mean_absolute_error(y_reg_test, reg_preds))

def create_movie_input(features, idea):
    new_movie = pd.DataFrame([[0.0]*len(features)], columns=features)
    for key in ["budget","runtime","popularity","vote_average","vote_count","year","month"]:
        new_movie.loc[0, key] = idea[key]
    for g in idea["genres"]:
        if g in new_movie.columns:
            new_movie.loc[0, g] = 1
    for comp in idea["companies"]:
        col_name = "comp_" + comp
        if col_name in new_movie.columns:
            new_movie.loc[0, col_name] = 1
    return new_movie

movie_ideas = [
    {
        "title": "The Last Signal",
        "budget": 8000000,
        "runtime": 95,
        "popularity": 35,
        "vote_average": 6.2,
        "vote_count": 200,
        "year": 2025,
        "month": 10,
        "genres": ["Horror","Mystery"],
        "companies": ["Blumhouse Productions"]
    },
    {
        "title": "Orbit Fall",
        "budget": 45000000,
        "runtime": 110,
        "popularity": 50,
        "vote_average": 6.8,
        "vote_count": 500,
        "year": 2025,
        "month": 7,
        "genres": ["Action","Science Fiction"],
        "companies": ["Universal Pictures"]
    },
    {
        "title": "Deepfake Justice",
        "budget": 30000000,
        "runtime": 105,
        "popularity": 45,
        "vote_average": 7.0,
        "vote_count": 400,
        "year": 2025,
        "month": 9,
        "genres": ["Thriller","Crime"],
        "companies": ["Warner Bros."]
    }
]

results = []

for idea in movie_ideas:
    new_movie = create_movie_input(X.columns, idea)
    success = clf_model.predict(new_movie)[0]
    roi = np.expm1(reg_model.predict(new_movie)[0])
    results.append({
        "title": idea["title"],
        "predicted_success": success,
        "predicted_roi": round(roi, 2)
    })

results_df = pd.DataFrame(results)
print(results_df)