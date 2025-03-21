import pandas as pd

import fire


def feat_eng(
    csv_in: str,
    csv_out: str,
) -> None:
    """Add new features.

    Features:
    - delta_t - DateTime_N+1 - DateTime_N

    Args:
        csv_in: Input CSV file
        csv_out: Output CSV file

    Return:
        None
    """
    print(f"Feature engineering for: {csv_in}")
    df = pd.read_csv(csv_in)

    df["delta_t"] = df["DateTime"].diff()

    # Columns from Damian
    df['mud_temp_diff'] = df['Mud Temp Out (°F)'] - df['Mud Temp In (°F)']
    df['ML_mud_temp_diff'] = df['ML Mud Temp OUT (°F)'] - df['ML Mud Temp IN (°F)']
    # df['bottoms_up_time'] = df['Depth(ft)'] / df['Annular Velocity (ft/min)']  # Can't use, I removed annular vel.
    df['delta_time_on_bottom'] = df["Time On Bottom (hr)"].diff()
    df['delta_circulating_time'] = df["Circulating Hrs (hr)"].diff()
    df['delta_time_on_job'] = df["Time On Job (hr)"].diff()
    df["delta_mud_volume"] = df["Mud Volume (bbl)"].diff()

    print("Saving CSV to", csv_out)
    df.to_csv(csv_out, index=False)


if __name__ == "__main__":
    fire.Fire(feat_eng)
