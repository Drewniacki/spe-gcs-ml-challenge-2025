import fire
import pandas as pd


def split_xy(
    csv_in: str,
    csv_x: str,
    csv_y: str,
    target_col: str = "Bttm Pipe Temp (°F)",
) -> None:
    """Splits data into two dataframes with features and target variable.

    Args:
        csv_in: Input CSV file
        csv_x: Output CSV file with features
        csv_y: Output CSV file with target
        target_col: Target variable name (default: "Bttm Pipe Temp (°F)")

    Return:
        None
    """
    print("Splitting features from target in:", csv_in)
    df = pd.read_csv(csv_in)

    x = df.drop(columns=target_col)
    x.to_csv(csv_x, index=False)

    y = df[[target_col]]
    y.to_csv(csv_y, index=False)

    print("Saved:", csv_x, csv_y)


if __name__ == "__main__":
    fire.Fire(split_xy)
