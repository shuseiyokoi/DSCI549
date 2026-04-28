from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
import time
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your fine-tuned model
MODEL = "gpt-3.5-turbo"

INPUT_PATH = "../data/tmdb_movies.csv"
OUTPUT_PATH = "../data/tmdb_movies_with_llm_rating_base_model.csv"


def extract_integer(text):
    """
    Extract integer from model output and force it into 0-100 range.
    """
    match = re.search(r"\d+", text)

    if not match:
        return None

    rating = int(match.group())

    # Force valid range
    rating = max(0, min(100, rating))

    return rating


def rate_overview(overview):
    """
    Ask fine-tuned model to rate movie overview.
    Output should be integer only.
    """

    if pd.isna(overview) or str(overview).strip() == "":
        return None

    prompt = f"""
Rate this movie overview from 0 to 100 based on its potential ROI and commercial performance.

Rules:
- Output ONLY one integer.
- No explanation.
- No words.
- No punctuation.
- Minimum is 0.
- Maximum is 100.

Overview:
{overview}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie ROI evaluator. Based on overview of movies, you would give rating 0 to 100. You only output one integer from 0 to 100.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=5,
        )

        output = response.choices[0].message.content.strip()
        rating = extract_integer(output)

        return rating

    except Exception as e:
        print(f"Error: {e}")
        return None


# Load data
df = pd.read_csv(INPUT_PATH)

# Make sure overview column exists
if "overview" not in df.columns:
    raise ValueError("Column 'overview' not found in dataset.")

# Optional: remove rows without overview
df = df.dropna(subset=["overview"]).copy()

# Create empty rating column if not exists
if "overview_rating" not in df.columns:
    df["overview_rating"] = None

# df_temp = df.head(5)

# Rate each overview
for i, row in df.iterrows():
    # Skip if already rated
    if pd.notna(row["overview_rating"]):
        continue

    overview = row["overview"]
    rating = rate_overview(overview)

    df.at[i, "overview_rating"] = rating

    print(f"{i}: overview = {overview[:60]}...")  # Print first 60 chars of overview
    print(f"{i}: rating = {rating}")

    # Save progress every 20 rows
    if i % 20 == 0:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Progress saved to {OUTPUT_PATH}")

    # Small delay to avoid rate limit
    time.sleep(0.3)

# Final save
df.to_csv(OUTPUT_PATH, index=False)

print("Done!")
print(f"Saved new dataset to {OUTPUT_PATH}")
