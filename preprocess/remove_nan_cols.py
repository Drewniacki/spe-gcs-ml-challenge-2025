import fire
import numpy as np
import pandas as pd


def get_nan_cols(df: pd.DataFrame) -> list[str]:
    """Get columns with NaNs.

    NaN columns are detected by checking the cross-correlation matrix.
    NaNs in this matrix are due to:
    - constant columns
    - NaNs in columns
    Both are considered as NaNs.
    """
    # Constant values
    corr_matrix = df.corr()
    col0 = corr_matrix.columns[0]
    const_cols = corr_matrix[col0].loc[corr_matrix[col0].isna()].index.tolist()

    # Remaining NaN values
    nan_value = -999.25
    df = df.where(df != nan_value, np.nan)
    nan_cols = df.columns[df.isna().any()].tolist()

    cols = const_cols + nan_cols
    return cols


def remove_nan_cols(*csv_files) -> None:
    """Overwrite input CSV files with new ones without constant/NaN columns."""
    # Get constant columns
    const_cols = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        const_cols.extend(get_nan_cols(df))

    # Remove duplicates
    const_cols = list(set(const_cols))
    print("Columns to be removed:", const_cols)

    # Overwrite CSV
    for csv in csv_files:
        df = pd.read_csv(csv)
        df = df.drop(columns=const_cols, errors="ignore")
        df.to_csv(csv, index=False)


if __name__ == "__main__":
    fire.Fire(remove_nan_cols)
