import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/tmdb_movies.csv")

df = df[df["budget"] > 1000]
df = df[df["revenue"] > 1000]
df = df[df["runtime"] > 40]

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])

df["roi"] = df["revenue"] / df["budget"]
df["roi"] = df["roi"].clip(upper=10)

df["success"] = (df["roi"] > 2).astype(int)

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

df = pd.concat([df, genre_df], axis=1)

features = [
    "budget",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
    "year",
    "month"
] + list(genre_df.columns)

X = df[features]
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

new_movie = pd.DataFrame(columns=X.columns)

new_movie.loc[0] = 0

new_movie["budget"] = 40000000
new_movie["runtime"] = 110
new_movie["popularity"] = 40
new_movie["vote_average"] = 6.5
new_movie["vote_count"] = 500
new_movie["year"] = 2025
new_movie["month"] = 7

for genre in ["Action", "Adventure"]:
    if genre in new_movie.columns:
        new_movie[genre] = 1

prediction = model.predict(new_movie)

print("Prediction:", prediction[0])