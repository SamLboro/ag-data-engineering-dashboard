# src/ingestion/fetch_prices.py
"""
Fetch commodity price data from Alpha Vantage.
"""
import os
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib import response

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_corn_prices(
    interval: str = "monthly",
    years_back: int = 5,
    save_path: Path = Path("../data/raw/"),
) -> pl.DataFrame:
    """
    Fetch corn price data from Alpha Vantage API.

    Args:
        interval: Time interval for the data (e.g., "monthly").
        years_back: Number of years back to fetch data for.
        save_path: Path to save the raw CSV data.

    Returns:
        A Polars DataFrame containing the fetched data.

    Raises:
        ValueError: If API key not found or invalid response
        requests.RequestException: If API request fails
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key for Alpha Vantage not found in environment variables."
        )

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CORN",
        "interval": interval,
        "datatype": "csv",
        "apikey": api_key,
    }

    print("Fetching corn price data from Alpha Vantage...")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse CSV
        df = pl.read_csv(
            StringIO(response.text),
            try_parse_dates=True,
            null_values=[".", "", "NA", "N/A", "null"],
        )

        print("Data fetched successfully.")

        # Clean nulls
        df_clean = df.drop_nulls()

        # Rename timestamp -> date for clarity
        df_clean = df_clean.rename({"timestamp": "date"})

        # Filter to recent years
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        df_filtered = df_clean.filter(pl.col("date") >= cutoff_date)

        # Sort by date
        df_filtered = df_filtered.sort("date")

        # # Save outputs
        # save_path.mkdir(parents=True, exist_ok=True)

        # csv_path = save_path / f"corn_{interval}_{years_back}yr.csv"
        # parquet_path = save_path / f"corn_{interval}_{years_back}yr.parquet"

        # df_filtered.write_csv(csv_path)
        # df_filtered.write_parquet(parquet_path)

        # print(f"Saved CSV: {csv_path}")
        # print(f"Saved Parquet: {parquet_path}")

        return df_filtered

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    # Allow running as script for testing
    df = fetch_corn_prices(interval="monthly", years_back=5)
    print(df)
