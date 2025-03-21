import torch


def save_model(
    epoch,
    model,
    optimizer,
    loss,
    scaler,
    num_features,
    checkpoint_path,
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "scaler.mean": scaler.mean,
        "scaler.std": scaler.std,
        "input_size": num_features,
    }, checkpoint_path)
