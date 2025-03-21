import os
import pandas as pd
import re
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_list_of_matching_models(criteria: list[str]) -> list[str]:
    """
    Get a list of model folder names that match all given criteria.

    :param criteria: A list of substrings that must be present as standalone segments in the folder name.
    :return: A list of matching folder names.
    """
    # Get all folders in the runs directory
    runs_dir = 'runs'
    all_runs_folders = [f for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]

    # Ensure each criterion appears as a standalone segment
    def matches_all_criteria(folder: str) -> bool:
        return all(re.search(rf'(^|[_-]){re.escape(c)}([_-]|$)', folder) for c in criteria)

    # Filter folders that match all criteria
    matching_folders = [folder for folder in all_runs_folders if matches_all_criteria(folder)]

    return matching_folders

def load_blind_prediction_csv(model: str, phase: int, hole: int) -> pd.DataFrame:
    blind_prediction_csv_name = f"phase{phase}_blind_test_predictions_{hole}.csv"
    blind_prediction_csv_path = Path(f'runs/{model}/0/{blind_prediction_csv_name}')
    blind_prediction_df = pd.read_csv(blind_prediction_csv_path)
    return blind_prediction_df

def load_blind_prediction_depth(phase: int, hole: int) -> pd.DataFrame:
    blind_prediction_depth_path = Path(f"data/raw/phase_{phase}/phase{phase}_blind_test_predictions_{hole}.csv")
    blind_prediction_depth_df = pd.read_csv(blind_prediction_depth_path)
    blind_prediction_depth_df = blind_prediction_depth_df[['Depth(ft)']]
    return blind_prediction_depth_df

def load_validation_depth_truth(phase: int, hole: int) -> pd.DataFrame:
    if phase==1:
        validation_depth_path = Path(f"data/formatted/phase_{phase}/Training_dataset_{hole}.csv")
    elif phase==2:
        validation_depth_path = Path(f"data/formatted/phase_{phase}/FineTune_Train_dataset_{hole}.csv")
    else:
        raise ValueError("phase must be 1 or 2")

    validation_depth_df = pd.read_csv(validation_depth_path)
    validation_depth_df = validation_depth_df[['Depth(ft)', 'Bttm Pipe Temp (°F)']]
    return validation_depth_df

def load_validation_csv(model: str, phase: int, hole: int) -> pd.DataFrame:
    validation_csv_name = f"validation_phase_{phase}_well_{hole}.csv"
    validation_csv_path = Path(f'runs/{model}/0/{validation_csv_name}')
    validation_df = pd.read_csv(validation_csv_path)
    validation_df = validation_df[['Depth(ft)', 'Bttm Pipe Temp (°F) - predicted']]
    return validation_df


    
if __name__ == "__main__":
    
    PHASE2_MAE_LIMIT = 15
    
    # look for models
    criteria_list = [
        ['p1trn0', 'p2trnNone', 'test16'], # phase 1, well 1
        ['p1trn1', 'p2trnNone', 'test16'], # phase 1, well 2
        ['p1trn2', 'p2trnNone', 'test16'], # phase 1, well 3
        ['p1trnNone', 'p2trn0', 'test16'], # phase 2, well 1
        ['p1trnNone', 'p2trn1', 'test16'], # phase 2, well 1
    ]

    matching_models = []
    for criteria in criteria_list:
        models = get_list_of_matching_models(criteria)
        print(f"{criteria=}")
        print(f"{len(models)} models found")
        print()
        matching_models.extend(models)

    print(f"{len(matching_models)} models found in total")
    print()
    
    # iterate through phases and wells to produce final submission
    phases_wells = {1: [1,2,3], 2: [1,2]}
    for phase, well_list in phases_wells.items():
        for well in well_list:
            
            # ------------------------
            # DATA LOADING
            # load predictions for selected models
            predictions = load_blind_prediction_depth(phase,well)
            for model in tqdm(matching_models, desc=f"Loading predictions for phase {phase}, well {well}"):
                single_prediction = load_blind_prediction_csv(model, phase=phase, hole=well)
                predictions = pd.merge(
                    predictions,
                    single_prediction.rename(columns={"Bttm Pipe Temp (°F) - predicted": str(model)}),
                    on='Depth(ft)',
                    how='left'
                )
                
            # for phase 2, load also validations
            if phase==2:
                validations = load_validation_depth_truth(phase,well)
                for model in tqdm(matching_models, desc=f"Loading validations for phase {phase}, well {well}"):
                    single_validation = load_validation_csv(model, phase=phase, hole=well)
                    validations = pd.merge(
                        validations,
                        single_validation.rename(columns={"Bttm Pipe Temp (°F) - predicted": str(model)}),
                        on='Depth(ft)',
                        how='left'
                    )
            
            # ------------------------
            # MODEL SELECTION
            
            # for phase 1, select equal number of models from each well
            if phase==1:
                model_selection = []
                for well_trained in ['p1trn0', 'p1trn1', 'p1trn2']:
                    # select all models trained on a given well from phase 1
                    well_models = [model for model in matching_models if well_trained in model]
                    # select random subset of 37 models
                    sample_length = min(37, len(well_models))
                    model_selection.extend(random.sample(well_models, sample_length))
            
            # for phase 2 select models based on validation score
            elif phase==2:
                ground_truth_col = "Bttm Pipe Temp (°F)"
                mae_results = [(col, mean_absolute_error(validations[ground_truth_col], validations[col])) for col in matching_models] # Compute MAE for each model validation
                mae_df = pd.DataFrame(mae_results, columns=["Model", "MAE"]).sort_values(by="MAE")

                model_selection = mae_df[mae_df['MAE']<=PHASE2_MAE_LIMIT]
                model_selection = model_selection["Model"].tolist()
            
            # save selected model report to file
            output_file_basename = f"phase{phase}_blind_test_predictions_{well}"
            models_report_file = Path(f"predictions/{output_file_basename}.txt")
            with open(models_report_file, "w") as f:
                f.write(f"predicting phase {phase}, well {well}\n")
                f.write(f"total number of models selected for prediction: {len(model_selection)}\n")
                f.write("that includes:\n\n")

                for well_trained in ['p1trn0', 'p1trn1', 'p1trn2', 'p2trn0', 'p2trn1']:
                    count = len([model for model in model_selection if well_trained in model])
                    f.write(f"- {well_trained}: {count} models\n")
                
                f.write("\nmodels selected for prediction:\n")
                f.write("\n".join(model_selection))
            print(f"used models report saved: {models_report_file}")
            
            # ------------------------
            # PRODUCE GRAPH
            
            graph_file = Path(f"predictions/{output_file_basename}.png")
            
            # Compute statistics
            for percentile in [11,41,50,59,89]:
                predictions[f"p{percentile}"] = predictions[model_selection].quantile(percentile/100, axis=1)
            predictions["mean"] = predictions[model_selection].mean(axis=1)
            predictions["min"] = predictions[model_selection].min(axis=1)
            predictions["max"] = predictions[model_selection].max(axis=1)

            # Plot
            plt.figure(figsize=(6, 10))
            plt.plot(predictions["p50"], predictions["Depth(ft)"], label=f"P50 - final prediction", color="blue", linestyle="dashed")
            plt.plot(predictions["mean"], predictions["Depth(ft)"], label=f"Mean", color="red", linestyle="dotted")

            # Fill area between procentiles
            plt.fill_betweenx(predictions["Depth(ft)"], predictions["min"], predictions["max"], color="blue", alpha=0.1, label="Min-Max")
            plt.fill_betweenx(predictions["Depth(ft)"], predictions["p11"], predictions["p89"], color="blue", alpha=0.2, label="P11-P89")
            plt.fill_betweenx(predictions["Depth(ft)"], predictions["p41"], predictions["p59"], color="blue", alpha=0.4, label="P41-P59")

            # scale graph
            min_val = predictions["p11"].min()
            min_val = round(min_val * 0.8 / 5) * 5
            max_val = predictions["p89"].max()
            max_val = round(max_val * 1.2 / 5) * 5
            plt.xlim(min_val, max_val)

            # Formatting
            plt.gca().invert_yaxis()  # Invert Depth axis
            plt.xlabel("Bottom Hole Circulating Temperature (°F)")
            plt.ylabel("Depth (ft)")
            plt.legend()
            plt.title(f"Phase {phase}, well {well} - prediction based on {len(model_selection)} models")
            plt.grid(True)

            plt.savefig(graph_file)
            print(f"graph of predictions saved: {graph_file}")
            
            # ------------------------
            # SAVE PREDICTION FILE
            prediction_file = Path(f"predictions/{output_file_basename}.csv")
            output_df = predictions[['Depth(ft)', "p50"]]
            output_df = output_df.rename(columns={"p50": "Bttm Pipe Temp (°F) - predicted"})
            output_df.to_csv(prediction_file, index=False)
            print(f"final prediction saved: {prediction_file}")
            