from datetime import datetime

import fire
import pandas as pd


def dt_to_int(
    csv_in: str,
    csv_out: str,
    dt_col: str,
    dt_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Convert datetime column to int data type.

    Args:
        csv_in: Input CSV file
        csv_out: Output CSV file
        dt_col: Datetime column name
        dt_fmt: Datetime format

    Return:
        None
    """
    print("Converting datetime strings to ints...")

    def dt_to_str_item(x):
        dt = datetime.strptime(x, dt_fmt)
        return int(dt.timestamp())

    df = pd.read_csv(csv_in)
    df[dt_col] = df[dt_col].map(dt_to_str_item)

    print("Saving CSV to", csv_out)
    df.to_csv(csv_out, index=False)


if __name__ == "__main__":
    fire.Fire(dt_to_int)
