"""
Bulk fetch and process Iowa weather data from NOAA Climate Data Online API.

This module provides functionality to fetch historical weather data (temperature,
precipitation) for Iowa from NOAA's GHCND dataset. It includes concurrent fetching
with retry logic, checkpointing for interrupted runs, and data cleaning utilities.

Typical usage:
    # Fetch all years 2020-2024
    python src/ingestion/fetch_weather_bulk.py

    # Fetch specific year
    python src/ingestion/fetch_weather_bulk.py --year 2020

    # Clean existing raw data
    python src/ingestion/fetch_weather_bulk.py --clean-only
"""

import json
import os
import time
from calendar import c
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from optparse import Option
from pathlib import Path
from threading import Lock
from tracemalloc import start
from typing import Any, Dict, List, Optional, Tuple
from wsgiref import headers

import click
import polars as pl
import requests
from bleach import clean
from dotenv import load_dotenv
from matplotlib.pylab import f

load_dotenv()


def fetch_batch_safe(
    offset: int, start_date: str, end_date: str, token: str, max_retries: int = 3
) -> Tuple[int, Optional[List[Dict[str, Any]]], int]:
    """
    Fetch a single batch of weather data from NOAA API with retry logic.

    Makes a request to NOAA's Climate Data Online API for a specific offset
    (pagination). Implements exponential backoff retry for 503 errors and
    handles other common error conditions.

    Args:
        offset: Starting record number for pagination (1-indexed).
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        token: NOAA API authentication token.
        max_retries: Maximum number of retry attempts for failed requests.
            Defaults to 3.

    Returns:
        A tuple containing:
            - offset (int): The offset that was requested.
            - results (list[dict] | None): List of weather records, or None if failed.
            - batch_size (int): Number of records returned (0 if failed/empty).
    """
    url = (
        f"https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        f"?datasetid=GHCND"
        f"&locationid=FIPS:19"
        f"&startdate={start_date}"
        f"&enddate={end_date}"
        f"&datatypeid=TMAX,TMIN,PRCP"
        f"&units=standard"
        f"&limit=1000"
        f"&offset={offset}"
    )

    headers = {"token": token}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=120)

            if response.status_code == 503:
                wait_time = 2**attempt
                print(f"503 at offset {offset}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if response.status_code == 502:
                return (offset, None, 0)

            if response.status_code != 200:
                return (offset, None, 0)

            data = response.json()

            if "results" not in data:
                return (offset, [], 0)

            results = data["results"]
            return (offset, results, len(results))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return (offset, None, 0)

    return (offset, None, 0)


def fetch_concurrent_sliding_window(
    start_date: str,
    end_date: str,
    max_workers: int = 4,
) -> Optional[pl.DataFrame]:
    """
    Fetch weather data using concurrent requests with sliding window pattern.

    Uses a thread pool to fetch multiple batches concurrently while respecting
    NOAA's rate limit (5 requests/second). Implements a sliding window where
    new requests are submitted as previous ones complete, maintaining a constant
    number of in-flight requests.

    Args:
        start_date: Start date in YYYY-MM-DD format (e.g., '2020-01-01').
        end_date: End date in YYYY-MM-DD format (e.g., '2020-12-31').
        max_workers: Maximum number of concurrent HTTP requests. Defaults to 4
            to stay safely under NOAA's 5 requests/second limit.

    Returns:
        Polars DataFrame containing weather records with columns:
            - date (str): ISO datetime string
            - datatype (str): 'TMAX', 'TMIN', or 'PRCP'
            - station (str): Weather station ID
            - attributes (str): Data quality attributes
            - value (float): Measured value
        Returns None if fetch fails or no data available.

    Raises:
        ValueError: If NOAA_TOKEN environment variable not found.
    """
    NOAA_TOKEN = os.getenv("NOAA_TOKEN")

    if not NOAA_TOKEN:
        raise ValueError("NOAA_TOKEN environment variable not found.")

    print(f"Fetching Iowa weather data from {start_date} to {end_date}...")

    all_results: List[Dict[str, Any]] = []
    next_offset = 1
    empty_batches = 0
    max_empty = 3

    results_lock = Lock()
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active_futures: Dict[Any, int] = {}

        # Submit initial batch of requests
        for i in range(max_workers):
            offset = next_offset
            future = executor.submit(
                fetch_batch_safe,
                offset,
                start_date=start_date,
                end_date=end_date,
                token=NOAA_TOKEN,
            )
            active_futures[future] = offset
            next_offset += 1000

        # Process completions with sliding window
        while active_futures:
            # Wait for at least one future to complete
            done, pending = wait(
                active_futures.keys(), return_when=FIRST_COMPLETED, timeout=120
            )

            if not done:
                print("Timeout waiting for requests to complete. Exiting.")
                break

            # Process all completed futures
            for future in done:
                offset = active_futures.pop(future)

                try:
                    fetch_offset, results, batch_size = future.result()

                    if results is None:
                        empty_batches += 1
                        print(f"Offset {fetch_offset:,} failed.")

                        if empty_batches >= max_empty:
                            print(f"Stopping after max failures")
                            # Cancel remaining futures
                            for f in active_futures.keys():
                                f.cancel()
                            active_futures.clear()
                            break

                    elif batch_size == 0:
                        empty_batches += 1
                        print(f"Offset {fetch_offset:,}: Empty")

                        if empty_batches >= max_empty:
                            print(f"End of data: {empty_batches} empty batches.")
                            for f in active_futures.keys():
                                f.cancel()
                            active_futures.clear()
                            break

                    else:
                        # Success - reset empty counter
                        empty_batches = 0

                        with results_lock:
                            all_results.extend(results)

                        print(
                            f"Offset {fetch_offset:,}: {batch_size} records "
                            f"(total: {len(all_results):,})"
                        )

                        # Submit next batch to sliding window
                        if next_offset < 500000:
                            new_future = executor.submit(
                                fetch_batch_safe,
                                next_offset,
                                start_date,
                                end_date,
                                NOAA_TOKEN,
                            )
                            active_futures[new_future] = next_offset
                            next_offset += 1000

                        time.sleep(0.22)  # Rate limiting

                except Exception as e:
                    print(f"Error processing offset {offset}: {e}")
                    empty_batches += 1

            # Check stopping condition
            if empty_batches >= max_empty:
                break

    elapsed = time.monotonic() - start_time

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total records: {len(all_results):,}")
    print(f"Speed: {len(all_results) / elapsed:.0f} records/sec")
    print(f"{'='*60}")

    if all_results:
        return pl.DataFrame(all_results)
    else:
        return None


# Checkpoint mgmt
def save_checkpoint(
    year: int, df: pl.DataFrame, checkpoint_dir: str = "checkpoints"
) -> None:
    """
    Save progress checkpoint for a year's weather data fetch.

    Creates both a Parquet file with the data and a JSON metadata file
    for tracking fetch progress. Useful for resuming interrupted fetches.

    Args:
        year: Year being fetched (e.g., 2020).
        df: Polars DataFrame containing records fetched so far.
        checkpoint_dir: Directory to save checkpoint files. Defaults to 'checkpoints'.

    Returns:
        None
    """

    checkpoint_path = Path(checkpoint_dir) / f"weather_{year}_checkpoint.parquet"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(checkpoint_path)

    # Save metadata
    meta_path = Path(checkpoint_dir) / f"weather_{year}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "year": year,
                "records": len(df),
                "timestamp": datetime.now().isoformat(),
                "completed": False,
            },
            f,
        )

    print("Checkpoint saved.")


def load_checkpoint(
    year: int,
    checkpoint_dir: str = "checkpoints",
) -> Optional[pl.DataFrame]:
    """
    Load progress checkpoint for a year if it exists.

    Args:
        year: Year to load checkpoint for (e.g., 2020).
        checkpoint_dir: Directory containing checkpoint files. Defaults to 'checkpoints'.

    Returns:
        Polars DataFrame with previously fetched records, or None if no checkpoint exists.
    """

    checkpoint_path = Path(checkpoint_dir) / f"weather_{year}_checkpoint.parquet"

    if checkpoint_path.exists():
        return pl.read_parquet(checkpoint_path)
    return None


def clear_checkpoint(year: int, checkpoint_dir: str = "checkpoints") -> None:
    """
    Delete checkpoint files after successful fetch completion.

    Args:
        year: Year to clear checkpoint for (e.g., 2020).
        checkpoint_dir: Directory containing checkpoint files. Defaults to 'checkpoints'.

    Returns:
        None
    """

    checkpoint_path = Path(checkpoint_dir) / f"weather_{year}_checkpoint.parquet"
    meta_path = Path(checkpoint_dir) / f"weather_{year}_meta.json"

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    print("Checkpoint cleared.")


def fetch_year_with_checkpoint(
    year: int,
    data_dir: str = "C:\\Users\\samb2\\Documents\\GitHub\\ag-data-engineering-dashboard\\data\\raw",
    checkpoint_dir: str = "checkpoints",
    max_workers: int = 4,
) -> Optional[pl.DataFrame]:
    """
    Fetch one year of Iowa weather data with automatic checkpoint support.

    Fetches daily weather observations (TMAX, TMIN, PRCP) for all Iowa weather
    stations for the specified year. Supports resuming from checkpoints if
    interrupted. Checks for existing completed files to avoid re-fetching.

    Args:
        year: Calendar year to fetch (e.g., 2020). Must be valid year.
        data_dir: Directory to save final Parquet files. Defaults to 'data/raw'.
        checkpoint_dir: Directory for checkpoint files during fetch. Defaults to 'checkpoints'.
        max_workers: Number of concurrent HTTP requests. Defaults to 4.

    Returns:
        Polars DataFrame containing weather records for the year, with columns:
            - date (str): ISO datetime string
            - datatype (str): Measurement type ('TMAX', 'TMIN', 'PRCP')
            - station (str): GHCND station identifier
            - attributes (str): Quality control attributes
            - value (float): Measured value
        Returns None if fetch fails or user declines re-fetch of existing data.

    Raises:
        KeyboardInterrupt: Propagated from fetch function if user interrupts.
        Exception: Any unexpected errors during fetch.
    """
    output_path = Path(data_dir) / f"iowa_weather_{year}_raw.parquet"

    # Check if already completed
    if output_path.exists():
        print(f"Data for year {year} already exists at {output_path}.")
        df_existing = pl.read_parquet(output_path)
        print(f"Records: {len(df_existing):,}")

        return df_existing

    # Check for checkpoint
    df_checkpoint = load_checkpoint(year, checkpoint_dir)
    if df_checkpoint is not None:
        print(f"Found checkpoint for year {year} with {len(df_checkpoint):,} records.")
        response = input(f"Resume from checkpoint? (Y/n): ").strip().lower()

        if response != "n":
            print("Resuming is not yet implemented - will fetch fresh")
            # TODO: Implement resume by checking last date in checkpoint

        clear_checkpoint(year, checkpoint_dir)

    # Define date range
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"\n{'='*60}")
    print(f"FETCHING: {year}")
    print(f"{'='*60}")

    # Fetch data
    try:
        df = fetch_concurrent_sliding_window(
            start_date, end_date, max_workers=max_workers
        )

        if df is None or len(df) == 0:
            print("No data fetched.")
            return None

        # Save final data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

        print(f"\n{year} COMPLETE")
        print(f"Records: {len(df):,}")

        # Clear any checkpoints
        clear_checkpoint(year, checkpoint_dir)

        return df

    except KeyboardInterrupt:
        print(f"\n\nInterrupted while fetching {year}")

        if df is not None and len(df) > 0:
            save_checkpoint(year, df, checkpoint_dir)
            print("Progress saved. Re-run to continue.")

        raise
    except Exception as e:
        print(f"\nError fetching {year}: {e}")
        return None


def fetch_all_years(
    start_year: int = 2020,
    end_year: int = 2024,
    data_dir: str = "C:\\Users\\samb2\\Documents\\GitHub\\ag-data-engineering-dashboard\\data\\raw",
    max_workers: int = 4,
) -> Optional[pl.DataFrame]:
    """
    Fetch multiple consecutive years of Iowa weather data.

    Fetches weather data for a range of years sequentially, with automatic
    combining into a single dataset. Supports interruption and resumption,
    with individual year files preserved.

    Args:
        start_year: First year to fetch (inclusive). Defaults to 2020.
        end_year: Last year to fetch (inclusive). Defaults to 2024.
        data_dir: Directory to save Parquet files. Defaults to 'data/raw'.
        max_workers: Number of concurrent HTTP requests per year. Defaults to 4.

    Returns:
        Combined Polars DataFrame containing all years of weather data.
        Returns None if no data successfully fetched.

    Raises:
        KeyboardInterrupt: If user interrupts during fetch. Progress is saved.
    """
    print(f"\n{'='*60}")
    print(f"BULK WEATHER FETCH: {start_year}-{end_year}")
    print(f"{'='*60}")

    all_dataframes: List[pl.DataFrame] = []
    overall_start = time.monotonic()

    for year in range(start_year, end_year + 1):
        try:
            df_year = fetch_year_with_checkpoint(
                year=year, data_dir=data_dir, max_workers=max_workers
            )

            if df_year is not None:
                all_dataframes.append(df_year)
                print(f"Year {year} fetched successfully.")
            else:
                print(f"Year {year} fetch failed.")

            # Pause between years
            if year < end_year:
                time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n\n⏸️  Interrupted during {year}")
            print(
                f"Years completed: {[start_year + i for i in range(len(all_dataframes))]}"
            )
            break

        except Exception as e:
            print(f"Unexpected error during {year}: {e}")
            print(f"Continuing to next year...")
            continue

    overall_elapsed = time.monotonic() - overall_start

    # combine all years
    if all_dataframes:
        print("Combining all years")

        df_combined = pl.concat(all_dataframes)

        combined_path = (
            Path(data_dir) / f"iowa_weather_{start_year}_{end_year}_raw.parquet"
        )
        df_combined.write_parquet(combined_path)

        print(f"\nBULK FETCH COMPLETE")
        print(f"{'='*60}")
        print(
            f"Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.1f} hours)"
        )
        print(f"Total records: {len(df_combined):,}")
        print(f"Years fetched: {len(all_dataframes)}")
        print(f"{'='*60}")

        return df_combined
    else:
        print("No data to combine")
        return None


# Data cleaning
def clean_weather_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform raw NOAA weather data from long to wide format.

    Converts long-format data (one row per station per datatype per day) to
    wide format (one row per day with columns for each datatype). Aggregates
    across all weather stations by taking daily means.

    Args:
        df: Raw weather DataFrame with columns:
            - date (str): ISO datetime string
            - datatype (str): 'TMAX', 'TMIN', or 'PRCP'
            - station (str): Station ID
            - value (float): Measured value

    Returns:
        Cleaned Polars DataFrame with columns:
            - date (date): Date object
            - temp_max_f (float): Average daily maximum temperature (°F)
            - temp_min_f (float): Average daily minimum temperature (°F)
            - precip_in (float): Average daily precipitation (inches), nulls filled with 0
        One row per unique date.

    Notes:
        - Aggregates multiple stations by taking mean per day/datatype
        - Missing precipitation values filled with 0.0 (no rain)
        - Temperature nulls preserved (may indicate data quality issues)
        - Typical reduction: ~230k raw records → ~365 daily records per year
    """
    print("\n" + "=" * 60)
    print("CLEANING WEATHER DATA")
    print("=" * 60)

    # Parse dates
    df = df.with_columns(
        [pl.col("date").str.strptime(pl.Date, "%Y-%m-%dT%H:%M:%S").alias("date")]
    )

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Aggregate across stations (average across date and datatype)
    df_agg = df.group_by(["date", "datatype"]).agg(
        [
            pl.col("value").mean().alias("value"),
            pl.col("station").count().alias("num_stations"),
        ]
    )

    # Pivot to wide format
    df_wide = df_agg.pivot(values="value", index="date", on="datatype")

    # Rename columns
    rename_map = {}
    if "TMAX" in df_wide.columns:
        rename_map["TMAX"] = "temp_max_f"
    if "TMIN" in df_wide.columns:
        rename_map["TMIN"] = "temp_min_f"
    if "PRCP" in df_wide.columns:
        rename_map["PRCP"] = "precip_in"

    df_clean = df_wide.rename(rename_map)

    # Sort by date
    df_clean = df_clean.sort("date")

    # Handle nulls
    df_clean = df_clean.with_columns([pl.col("precip_in").fill_null(0.0)])

    return df_clean


# CLI
@click.command(
    help="Fetch Iowa weather data from NOAA Climate Data Online",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Fetch specific year only (e.g., 2020)",
)
@click.option(
    "--start",
    type=int,
    default=2020,
    show_default=True,
    help="Start year for range",
)
@click.option(
    "--end",
    type=int,
    default=2024,
    show_default=True,
    help="End year for range",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of concurrent workers",
)
@click.option(
    "--data-dir",
    type=str,
    default="C:\\Users\\samb2\\Documents\\GitHub\\ag-data-engineering-dashboard\\data\\raw",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--clean-only",
    is_flag=True,
    default=False,
    help="Only clean existing raw data, skip fetch",
)
def main(
    year: Optional[int],
    start: int,
    end: int,
    workers: int,
    data_dir: str,
    clean_only: bool,
) -> None:
    """
    Main entry point for command-line interface using Click.

    Parses command-line arguments and executes appropriate fetch or clean operations.
    Provides multiple modes: fetch all years, fetch specific year, or clean existing data.

    Args:
        year: Fetch specific year only. If provided, overrides start/end.
        start: Start year for range (default: 2020).
        end: End year for range (default: 2024).
        workers: Number of concurrent HTTP requests (default: 4).
        data_dir: Output directory for files (default: 'data/raw').
        clean_only: If True, only clean existing data without fetching.

    Returns:
        None

    Exits:
        0: Success
        1: Error (missing token, no files found, etc.)

    Examples:
        # Fetch all years
        $ python fetch_weather_bulk.py

        # Fetch specific year
        $ python fetch_weather_bulk.py --year 2020

        # Fetch with more workers
        $ python fetch_weather_bulk.py --workers 6

        # Clean existing data
        $ python fetch_weather_bulk.py --clean-only
    """
    # Validate NOAA token
    if not clean_only:
        NOAA_TOKEN = os.getenv("NOAA_TOKEN")
        if not NOAA_TOKEN:
            click.echo("Error: NOAA_TOKEN environment variable not found.")
            raise click.Abort()

    try:
        if clean_only:
            # Clean existing raw data files
            click.echo("Cleaning existing raw data files...")
            data_path = Path(data_dir)

            raw_files = list(data_path.glob("iowa_weather_*_raw.parquet"))

            if not raw_files:
                click.echo("No raw data files found to clean.")
                return

            for raw_file in raw_files:
                df_raw = pl.read_parquet(raw_file)
                df_clean = clean_weather_data(df_raw)

                clean_file = (
                    raw_file.parent.parent
                    / "processed"
                    / raw_file.name.replace("_raw", "_clean")
                )
                clean_file.parent.mkdir(parents=True, exist_ok=True)
                df_clean.write_parquet(clean_file)

                click.echo(f"Cleaned data saved to {clean_file}")
        elif year:
            # Fetch specific year
            df = fetch_year_with_checkpoint(
                year=year,
                data_dir=data_dir,
                max_workers=workers,
            )

            if df is not None:
                click.echo("Fetch completed successfully.")
        else:
            # Fetch range of years
            df = fetch_all_years(
                start_year=start,
                end_year=end,
                data_dir=data_dir,
                max_workers=workers,
            )

            if df is not None:
                click.echo("Bulk fetch completed successfully.")

    except KeyboardInterrupt:
        click.echo("\n\nProcess interrupted by user.")
        click.echo("Progress saved, Re-run to continue.")
    except click.Abort:
        pass
    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
