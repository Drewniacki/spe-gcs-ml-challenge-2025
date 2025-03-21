import numpy as np
import torch

from models.load_model import load_model
from models.standardscaler import TorchStandardScaler
from config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    SEQ_LEN,
    USE_STATE_EVL,
)


def evaluate_checkpoint(checkpoint_path: str, x: np.ndarray) -> np.ndarray:
    """Runs model from checkpoint on input data x.

    Evaluation is run on CPU.
    The checkpoint should additionally contain fields `scaler.mean` and `scaler.std`.

    Args:
        checkpoint_path: Path to the checkpoint file
        x: Input data stored in numpy array, shape (num_items, num_features)

    Returns:
        Predicted target, shape (num_items, 1)
    """
    # Load model
    model = load_model(checkpoint_path, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, USE_STATE_EVL)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    scaler = TorchStandardScaler()
    scaler.mean = checkpoint["scaler.mean"]
    scaler.std = checkpoint["scaler.std"]
    scaler.mean = scaler.mean.to("cpu")
    scaler.std = scaler.std.to("cpu")

    y = evaluate_model(model, scaler, x)

    return y


def evaluate_model(model: torch.nn.Module, scaler: TorchStandardScaler, x: np.ndarray) -> np.ndarray:
    """Runs model on input data x.

    Evaluation is run on CPU.
    The checkpoint should additionally contain fields `scaler.mean` and `scaler.std`.

    Args:
        model: torch model
        scaler: standard scaler
        x: Input data stored in numpy array, shape (num_items, num_features)

    Returns:
        Predicted target, shape (num_items, 1)
    """
    x_t = torch.from_numpy(x.astype(np.float32))
    x_norm = scaler.transform(x_t)
    y, _ = model.process_sequence(x_norm)
    y = y.detach().numpy()

    return y
