from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import evaluate_checkpoint
from config import SEQ_LEN


NUM_INIT_ROWS = SEQ_LEN * 3


def predict_phase_1(checkpoint_path: str):
    x_init_paths = [
        "data/formatted/phase_1/Training_dataset_1_x.csv",
        "data/formatted/phase_1/Training_dataset_2_x.csv",
        "data/formatted/phase_1/Training_dataset_3_x.csv",
    ]
    x_test_paths = [
        "data/formatted/phase_1/Test_dataset_1.csv",
        "data/formatted/phase_1/Test_dataset_2.csv",
        "data/formatted/phase_1/Test_dataset_3.csv",
    ]
    target_csvs = [
        "data/raw/phase_1/phase1_blind_test_predictions_1.csv",
        "data/raw/phase_1/phase1_blind_test_predictions_2.csv",
        "data/raw/phase_1/phase1_blind_test_predictions_3.csv",
    ]

    for x_init_path, x_test_path, target_csv in zip(x_init_paths, x_test_paths, target_csvs):
        model_dir = str(Path(checkpoint_path).parent.parent)

        # Load initial measurements (for state stabilization)
        x_init = pd.read_csv(x_init_path, index_col=False).to_numpy()[-NUM_INIT_ROWS:]
        del x_init_path

        # Load test measurements
        x_test_df = pd.read_csv(x_test_path, index_col=False)
        x_test = x_test_df.to_numpy()
        print(f"{Path(target_csv).stem}: {x_test.shape=}")

        # Stack initial and test measurements
        x_test = np.vstack((x_init, x_test))

        del x_test_path
        del x_init

        # Get depth vector
        depth_col_name = "Depth(ft)"
        depth_col_num = np.argwhere(x_test_df.columns == depth_col_name).flatten()[0]
        depth = x_test[:, depth_col_num]
        del x_test_df
        del depth_col_name

        # Predict BHCT
        y_pred = evaluate_checkpoint(checkpoint_path, x_test)
        y_pred = y_pred[NUM_INIT_ROWS:]
        depth = depth[NUM_INIT_ROWS:]
        print(f"{Path(target_csv).stem}: {y_pred.shape=}")

        # Interpolate
        target_col = "Bttm Pipe Temp (°F) - predicted"
        depth_col = "Depth(ft)"
        target_df = pd.read_csv(target_csv, index_col=0)
        pred_df = pd.DataFrame(data=np.hstack((depth.reshape(-1, 1), y_pred)),
                               columns=[depth_col, target_col]).set_index(depth_col)

        # Perform linear interpolation on the 'Bttm Pipe Temp (°F) - predicted' values
        target_df['Bttm Pipe Temp (°F) - predicted'] = np.interp(
            target_df.index.values,
            pred_df.index.values,
            pred_df['Bttm Pipe Temp (°F) - predicted'].values
        )

        # Save predictions
        output_df_path = str(Path(model_dir) / Path(target_csv).name)
        target_df.to_csv(output_df_path)
        print(f"prediction saved: {output_df_path}")

        # Make plot
        title = Path(target_csv).stem
        output_fig_path = str(Path(model_dir) / Path(target_csv).stem) + ".png"
        plt.figure(figsize=(6, 12), dpi=100)
        plt.title(title)
        plt.plot(y_pred, depth, c="blue", alpha=0.7, label="Predictions")
        plt.ylabel("Depth [ft]")
        plt.xlabel("BHCT [F]")
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        plt.savefig(output_fig_path)
        plt.close()
        print(f"graph saved: {output_fig_path}")


if __name__ == "__main__":
    model = "P1-train012_test0-seq16-hid32-lay2-drp0.2-bch256-test300-ep500-4b80ea75"
    for split in range(3):
        checkpoint_path = f"runs/{model}/{split}/checkpoints/best_trn_model.pth"
        predict_phase_1(checkpoint_path)
