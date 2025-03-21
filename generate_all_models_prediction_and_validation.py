from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

from validation_predict import predict_validation, get_target_csv_name
from phase_1_predict import predict_phase_1
from phase_2_predict import predict_phase_2


if __name__ == "__main__":
    models = [d for d in os.listdir('runs') if os.path.isdir(os.path.join('runs', d))]
    
    i=0
    for model in models:
        
        start_time = time.time() # Start time of the iteration
        
        i += 1
        print("\n========================================")
        print(f"MODEL {i}/{len(models)}: {model}")
        checkpoint_path=Path(f"runs/{model}/0/checkpoints/best_trn_model.pth")
        
        # predict phase 2 validatiosn (for model selection)
        predict_validation(checkpoint_path, validation_phase=2, validation_well=0)
        predict_validation(checkpoint_path, validation_phase=2, validation_well=1)

        # Run prediction on test data
        predict_phase_1(checkpoint_path)
        predict_phase_2(checkpoint_path, well_no=0)
        predict_phase_2(checkpoint_path, well_no=1)
        
        end_time = time.time()  # End time of the iteration
        iteration_time = end_time - start_time  # Time taken for this iteration
        print(f"Iteration completed in: {iteration_time:.1f} seconds")