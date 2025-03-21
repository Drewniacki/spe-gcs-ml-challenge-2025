import fire
import numpy as np
import pandas as pd


def clean_data(csv_in: str, csv_out: str, max_nan: float = 0.01) -> None:
    """Clean data from the CSV and save a new one.

    This script does the following:
    1. Reads `csv_in`.
    2. Removes columns with the number of NaNs > `max_nan`.
    3. Removes remaining lines with any NaN values.
    4. Drops lines with delta_t <= 0.
    5. Save a new CSV to `csv_out`.

    Args:
        csv_in: Input CSV file
        csv_out: Output CSV file
        max_nan: Maximum share of NaNs in a column

    Return:
        None
    """
    df = pd.read_csv(csv_in)
    num_cols = df.index.size
    cols_to_drop = []

    # nan_value = -999.25
    # df = df.where(df != nan_value, np.nan)

    for col in df.columns:
        if df[col].isna().sum() / num_cols > max_nan:
            cols_to_drop.append(col)

    print("Dropping columns:", cols_to_drop)
    df = df.drop(columns=cols_to_drop)

    print("Dropping lines with NaNs...")
    size_before = df.index.size
    df = df.loc[~df.isna().any(axis=1)]
    size_after = df.index.size
    print("Lines dropped:", size_before - size_after)

    print("Dropping lines with same delta_t <= 0")
    df = df.loc[df["delta_t"] > 0]

    print("Saving CSV to", csv_out)
    df.to_csv(csv_out, index=False)


if __name__ == "__main__":
    fire.Fire(clean_data)
