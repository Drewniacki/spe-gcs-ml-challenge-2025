import torch

from .gru import GRU


def load_model(
    checkpoint_path: str,
    hidden_size: int,
    num_layers: int,
    seq_len: int,
    use_state_evl: bool,
    device: str | None = None,
) -> GRU:

    if device is None:
        device = "cpu"

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    input_size = checkpoint["input_size"]
    model = GRU(input_size, hidden_size, num_layers, seq_len, use_state_evl)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to("cpu")

    return model
