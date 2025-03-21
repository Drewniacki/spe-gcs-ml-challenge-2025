import os

from config import LOG_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR


def prepare_dirs(model_name: str, split: int) -> tuple[str, str, str]:
    """Creates log and checkpoint directories.

    Args:
        model_name: Name of the model
        split: KFold split number

    Returns:
        (model_dir, checkpoint_dir, tensorboard_dir)
    """
    model_dir = os.path.join(LOG_DIR, model_name, str(split))
    checkpoint_dir = os.path.join(model_dir, CHECKPOINT_DIR)
    tensorboard_dir = os.path.join(model_dir, TENSORBOARD_DIR)

    os.makedirs(checkpoint_dir)
    os.makedirs(tensorboard_dir)

    return (model_dir, checkpoint_dir, tensorboard_dir)
