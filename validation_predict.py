from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

from evaluate import evaluate_checkpoint


def get_target_csv_name(phase: int, well: int):
    return f"validation_phase_{phase}_well_{well+1}.csv"

def predict_validation(checkpoint_path: str, validation_phase: int, validation_well: int):
    if str(validation_phase) == str(1):
        x_path = f"data/formatted/phase_{validation_phase}/Training_dataset_{validation_well+1}_x.csv"
        y_path = f"data/formatted/phase_{validation_phase}/Training_dataset_{validation_well+1}_y.csv"
    elif str(validation_phase) == str(2):
        x_path = f"data/formatted/phase_{validation_phase}/FineTune_Train_dataset_{validation_well+1}_x.csv"
        y_path = f"data/formatted/phase_{validation_phase}/FineTune_Train_dataset_{validation_well+1}_y.csv"
    else:
        raise ValueError(f"Phase has to be set to 1 or 2, {validation_phase} received.")
    target_csv = get_target_csv_name(validation_phase, validation_well)

    model_dir = str(Path(checkpoint_path).parent.parent)

    # Load test measurements
    x_valid_df = pd.read_csv(x_path, index_col=False)
    x_valid = x_valid_df.to_numpy()
    print(f"{Path(target_csv).stem}: {x_valid.shape=}")
    del x_path

    # Get depth vector
    depth_col_name = "Depth(ft)"
    depth_col_num = np.argwhere(x_valid_df.columns == depth_col_name).flatten()[0]
    depth = x_valid[:, depth_col_num]
    del x_valid_df


    # Predict BHCT
    y_pred = evaluate_checkpoint(checkpoint_path, x_valid).ravel()
    print(f"{Path(target_csv).stem}: {y_pred.shape=}")
    
    # combine depth, predicted and ground truth
    target_df = pd.read_csv(y_path, index_col=False)
    ground_truth_column_name = target_df.columns[0]
    ground_truth = target_df[ground_truth_column_name].to_numpy()
    print(f"{Path(target_csv).stem}: {ground_truth.shape=}")
    target_df[depth_col_name] = depth
    predicted_column_name = 'Bttm Pipe Temp (°F) - predicted'
    target_df[predicted_column_name] = y_pred
    
    # reorder columns
    target_df = target_df[[depth_col_name, ground_truth_column_name, predicted_column_name]]
    del depth_col_name
    del ground_truth_column_name
    del predicted_column_name

    # Save predictions
    output_df_path = str(Path(model_dir) / Path(target_csv).name)
    target_df.to_csv(output_df_path, index=False)
    print(f"prediction saved: {output_df_path}")
    
    # calculate MAE
    mae_valid = np.mean(np.abs(y_pred - ground_truth))
    np.savetxt
    print(f"{mae_valid=}")

    # Make plot
    title = f"{Path(target_csv).stem}, MAE={mae_valid:.2f}"
    output_fig_path = str(Path(model_dir) / Path(target_csv).stem) + ".png"
    plt.figure(figsize=(6, 12), dpi=100)
    plt.title(title)
    plt.plot(y_pred, depth, c="red", alpha=0.7, label="Predictions")
    plt.plot(ground_truth, depth, c="black", label="Ground truth")
    plt.ylabel("Depth [ft]")
    plt.xlabel("BHCT [F]")
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(output_fig_path)
    plt.close()
    print(f"graph saved: {output_fig_path}")


if __name__ == "__main__":
    TEST_PHASE = 2
    TEST_HOLE = 0
    MODEL = "Krzysiek_p1trn0_p1tst0_p2trnNone_p2tstNone-seq16-hid16-bch128-test16-ep200-7ff38b3b"
    FORCE = True
    
    run_path = os.path.join('runs',MODEL,'0')
    target_csv_name = get_target_csv_name(TEST_PHASE, TEST_HOLE)
    target_csv_path = os.path.join(run_path,target_csv_name)
    if not os.path.exists(target_csv_path) or FORCE:
        print(f"generating {target_csv_path}")
        checkpoint_path = os.path.join(run_path, 'checkpoints', 'best_trn_model.pth')
        predict_validation(checkpoint_path, TEST_PHASE, TEST_HOLE)

