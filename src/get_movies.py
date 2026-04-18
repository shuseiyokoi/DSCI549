import requests
import pandas as pd
import time
from datetime import date
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
from datetime import timedelta
import os

tmdb_api_key = input("Enter TMDB API key: ")

DEFAULT_START_DATE = "2000-01-01"
END_DATE = "2025-12-31"


def get_movies(start_date=DEFAULT_START_DATE, end_date=END_DATE):
    tmdb_url = "https://api.themoviedb.org/3/discover/movie"
    all_movies = []
    page = 1

    while True:
        params = {
            "api_key": tmdb_api_key,
            "page": page,
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date,
            "sort_by": "primary_release_date.asc",
            "vote_count.gte": 1,
            "with_origin_country": "US",
        }

        try:
            response = requests.get(tmdb_url, params=params)
            response.raise_for_status()
            movie_data = response.json()
        except requests.exceptions.RequestException:
            break

        results = movie_data.get("results", [])
        if not results:
            break

        all_movies.extend(results)
        print(f"Fetched page {page}")

        total_pages = movie_data.get("total_pages", 1)
        if page >= total_pages or page >= 200:
            break

        page += 1

    return all_movies


def get_movie_info(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": tmdb_api_key, "append_to_response": "credits"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        info_get = response.json()
    except requests.exceptions.RequestException:
        return None

    if (info_get.get("budget") or 0) == 0 or (info_get.get("revenue") or 0) == 0:
        return None

    return {
        "tmdb_id": movie_id,
        "title": info_get.get("title"),
        "release_date": info_get.get("release_date"),
        "runtime": info_get.get("runtime"),
        "budget": info_get.get("budget") or 0,
        "revenue": info_get.get("revenue") or 0,
        "cast": [c["name"] for c in info_get.get("credits", {}).get("cast", [])[:10]],
        "crew": [c["name"] for c in info_get.get("credits", {}).get("crew", [])[:10]],
        "countries": [
            c["iso_3166_1"] for c in info_get.get("production_countries", [])
        ],
        "status": info_get.get("status"),
        "overview": info_get.get("overview"),
        "popularity": info_get.get("popularity"),
        "vote_average": info_get.get("vote_average"),
        "vote_count": info_get.get("vote_count"),
        "tagline": info_get.get("tagline"),
        "genres": [g["name"] for g in info_get.get("genres", [])],
        "production_companies": [
            c["name"] for c in info_get.get("production_companies", [])
        ],
    }


def get_resume_start_date(output_path):
    if not os.path.exists(output_path):
        return DEFAULT_START_DATE

    existing_df = pd.read_csv(output_path)

    if existing_df.empty or "release_date" not in existing_df.columns:
        return DEFAULT_START_DATE

    existing_df["release_date"] = pd.to_datetime(
        existing_df["release_date"], errors="coerce"
    )

    latest_date = existing_df["release_date"].max()

    if pd.isna(latest_date):
        return DEFAULT_START_DATE

    next_date = latest_date + timedelta(days=1)
    return next_date.strftime("%Y-%m-%d")


def main():
    output_path = "../data/tmdb_movies.csv"

    start_date = get_resume_start_date(output_path)
    print(f"Starting from: {start_date}")

    dataset = []

    for movie in get_movies(start_date=start_date, end_date=END_DATE):
        movie_id = movie["id"]
        details = get_movie_info(movie_id)
        time.sleep(0.0001)

        print(f"Processing movie {movie_id}")

        if details:
            dataset.append(details)

    if not dataset:
        print("No new movies found.")
        return pd.DataFrame()

    df = pd.DataFrame(dataset)

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Added {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    df = main()
