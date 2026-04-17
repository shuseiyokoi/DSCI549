## How to Run `get_movies.py`

`get_movies.py` is a Python script that collects movie details from TMDB and saves them to `data/tmdb_movies.csv`.

### What it does
1. Checks `data/tmdb_movies.csv` for the latest saved date
2. Starts fetching movie data from the next day
3. Appends the new records to `data/tmdb_movies.csv`

[get_movies.py lines 45–72](https://github.com/shuseiyokoi/DSCI549/blob/main/src/get_movies.py#L46-L49).

```python
        total_pages = movie_data.get("total_pages", 1)
        if page >= total_pages or page >= 5:
            break
```

You can control how much data to fetch by changing the iteration values in the script.  
Fetching 200 pages usually takes about 5 minutes.

### How to run it

1. Open your terminal and go to the `src` directory

 ```bash
 cd src
 ```

2.	Run the script

  ```bash
  python get_movies.py
  ```

3.	Enter your TMDB API key when prompted
  
  ```bash
  Enter TMDB API key:
  ```

4.	The script will fetch and save the new movie data automatically.
