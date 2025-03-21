import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4

from config import (
    SEQ_LEN,
    NUM_EPOCHS,
    TEST_SIZE,
    MAX_ABS_X,
    MAX_ABS_Y,
    BATCH_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    LR,
    NAME,
    SUP_NAME,
    USE_STATE_TRN,
    USE_STATE_EVL,
    P1_TEST_HOLE,
    P1_TRAIN_HOLE,
    P2_TEST_HOLE,
    P2_TRAIN_HOLE,
)
from count_params import count_params
from dataset import TemperatureDataset
from models.gru import GRU
from models.standardscaler import TorchStandardScaler
from models.save_model import save_model
from prepare_dirs import prepare_dirs
from evaluate import evaluate_checkpoint
from sampler import HoleShuffler
from phase_1_predict import predict_phase_1
from phase_2_predict import predict_phase_2
from validation_predict import predict_validation


# Model name
model_name = NAME + "-" + SUP_NAME + "-" + str(uuid4())[:8]
print("TRAINING:", model_name)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the data
x_p1_paths = [
    "data/formatted/phase_1/Training_dataset_1_x.csv",
    "data/formatted/phase_1/Training_dataset_2_x.csv",
    "data/formatted/phase_1/Training_dataset_3_x.csv",
]
y_p1_paths = [
    "data/formatted/phase_1/Training_dataset_1_y.csv",
    "data/formatted/phase_1/Training_dataset_2_y.csv",
    "data/formatted/phase_1/Training_dataset_3_y.csv",
]
x_p2_paths = [
    "data/formatted/phase_2/FineTune_Train_dataset_1_x.csv",
    "data/formatted/phase_2/FineTune_Train_dataset_2_x.csv",
]
y_p2_paths = [
    "data/formatted/phase_2/FineTune_Train_dataset_1_y.csv",
    "data/formatted/phase_2/FineTune_Train_dataset_2_y.csv",
]

def to_list(x: list | int | None) -> list[int]:
    if x is None:
        return []
    elif isinstance(x, int):
        x = [x]
    assert isinstance(x, list)
    return x

train_holes_p1 = to_list(P1_TRAIN_HOLE)
train_holes_p2 = to_list(P2_TRAIN_HOLE)

# We decided to train on a single well.
# I keep below as lists, because the rest of the code assumes so.
x_p1_paths = [x_p1_paths[i] for i in train_holes_p1]
y_p1_paths = [y_p1_paths[i] for i in train_holes_p1]
x_p2_paths = [x_p2_paths[i] for i in train_holes_p2]
y_p2_paths = [y_p2_paths[i] for i in train_holes_p2]

x_paths = x_p1_paths + x_p2_paths
y_paths = y_p1_paths + y_p2_paths

# Get depth column number
df = pd.read_csv(x_paths[0])
depth_col_name = "Depth(ft)"
depth_col_num = np.argwhere(df.columns == depth_col_name).flatten()[0]
del depth_col_name
del df

# Create dataset and splits
dataset = TemperatureDataset(x_paths, y_paths, SEQ_LEN)

if P1_TEST_HOLE is not None:
    if len(train_holes_p1) == 3:
        test_hole = P1_TEST_HOLE
    elif len(train_holes_p1) == 1:
        test_hole = 0
    else:
        raise ValueError("Training on 2 wells in Phase 1 is not supported. Pick 1 or 3.")
elif P2_TEST_HOLE is not None:
    if len(train_holes_p2) == 2:
        test_hole = P2_TEST_HOLE + len(train_holes_p1)  # Offset needed
    elif len(train_holes_p2) == 1:
        test_hole = 0 + len(train_holes_p1)  # Offset needed
    else:
        raise ValueError("Test well in Phase 2 not specified correctly")
else:
    raise ValueError("Test well chosen incorrectly")

train_index, test_index, ext_test_index = dataset.get_train_test_split(
    test_hole_num=test_hole,
    test_size=TEST_SIZE,
)

# Create log and checkpoint directories
model_dir, checkpoint_dir, tensorboard_dir = prepare_dirs(model_name, test_hole)

# Copy config to model directory
shutil.copy("config.py", os.path.join(model_dir, "config.py"))

# Get all training data to train the scaler
x_train, _ = dataset.get_selected(list(train_index))

# Train scaler
scaler = TorchStandardScaler()
scaler.fit(x_train.to(device))

# x_train is no longer needed, because x_train will be retrieved from DataLoader
del x_train

# Create PyTorch datasets and loaders
# Training data (loader used)
# sampler = RandomSampler(list(train_index))
# train_sampler = list(train_index)  # Alternative - just go through the ordered list
train_sampler = HoleShuffler(dataset, test_hole, TEST_SIZE)
drop_last = True if USE_STATE_TRN is True else False
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=drop_last)

# Get test data
x_test, y_test = dataset.get_selected(list(ext_test_index))

# Save depth and ground truth temperature, because they will be used in output plots
depth = x_test[:, depth_col_num].numpy()
ground_truth = y_test[:, 0].numpy()

# Save x_test in numpy, because it will be used to evaluate the model at the end
x_test_numpy = x_test.numpy()

# Send test data to GPU and scale x
x_test_norm = scaler.transform(x_test.to(device))
y_test = y_test.to(device)

# Initialize the model
hidden_size = HIDDEN_SIZE
num_layers = NUM_LAYERS
num_features = x_test_norm.size()[-1]  # Because its shape is (num_items, num_features)

# create the model
model = GRU(num_features, hidden_size, num_layers, SEQ_LEN, USE_STATE_EVL).to(device)

# Count parameters and print summary
count_params(model)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mae = nn.L1Loss()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=tensorboard_dir)

# Variables for checkpointing
best_val_loss = float("inf")
best_trn_loss = float("inf")

# Training loop
epochs = NUM_EPOCHS
final_loss = 0.
outputs = None
state = None

for epoch in range(epochs):
    # Train with backpropagation
    model.train()
    running_loss = 0.0

    for x_batch, y_batch, hole_start in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Reset state if beginning of a new drill
        if hole_start is True:
            print("NEW HOLE -> RESETTING STATE")
            state = None

        # Skip incomplete batches
        if x_batch.size()[0] != BATCH_SIZE:
            continue

        # Normalize x
        x_batch = scaler.transform(x_batch)

        # Add noise
        if MAX_ABS_X > 0:
            x_batch += (torch.rand_like(x_batch) - 0.5) * 2. * MAX_ABS_X
        if MAX_ABS_Y > 0:
            y_batch += (torch.rand_like(y_batch) - 0.5) * 2. * MAX_ABS_Y

        # Forward pass
        if USE_STATE_TRN:
            outputs, state = model(x_batch, state)
        else:
            outputs, state = model(x_batch, None)
        N = 1
        loss = criterion(outputs[:, -N:, :], y_batch[:, -N:, :])  # Compare only last N values

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = state.clone().detach()  # Needed to not calc. gradient multiple times over same tensor

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    final_loss = epoch_loss
    writer.add_scalar("Training Loss", epoch_loss, epoch)
    assert isinstance(state, torch.Tensor)
    print(
        f"{model_name} | Epoch {epoch + 1}/{epochs}, "
        f"Training Loss: {epoch_loss:.4f}, "
        f"Hidden state: {torch.mean(torch.abs(state)):.3f}"  # Should not be close to zero
    )

    if epoch_loss < best_trn_loss:
        best_trn_loss = epoch_loss
        save_model(epoch, model, optimizer, epoch_loss, scaler, num_features, f"{checkpoint_dir}/best_trn_model.pth")
        print(f"Saved best training model at epoch {epoch + 1}")

    # Evaluate on validation set (k-fold cross-validation)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        outputs, state_val = model.process_sequence(x_test_norm)
        loss = criterion(outputs[-TEST_SIZE:], y_test[-TEST_SIZE:])  # Calculate loss only on the test segment
        val_loss += loss.item()

    writer.add_scalar("Validation Loss", val_loss, epoch)
    assert isinstance(state_val, torch.Tensor)
    print(
        f"{model_name} | Epoch {epoch + 1}/{epochs}, "
        f"Validation Loss: {val_loss:.4f}, "
        f"Hidden state: {torch.mean(torch.abs(state_val)):.3f}"  # Should not be close to zero
    )

    # Other metrics
    writer.add_scalar("Validation MAE", mae(outputs[-TEST_SIZE:], y_test[-TEST_SIZE:]), epoch)

# Save the final model checkpoint
save_model(NUM_EPOCHS, model, optimizer, final_loss, scaler, num_features, f"{checkpoint_dir}/final_model.pth")
print("Saved final model")

# Plot test predictions vs. ground truth
print("Evaluating on test data...")
pred_best_trn = evaluate_checkpoint(
    checkpoint_path=f"{checkpoint_dir}/best_trn_model.pth",
    x=x_test_numpy,
)

# Calculate MAE on the test segment
# mae_best_test = np.mean(np.abs(pred_best[-TEST_SIZE:] - ground_truth[-TEST_SIZE:]))
mae_best_trn = np.mean(np.abs(pred_best_trn[-TEST_SIZE:] - ground_truth[-TEST_SIZE:]))

# Get the depth at which the test segment begins
test_depth = x_test_numpy[-TEST_SIZE, depth_col_num]

phase = None
hole = None
if P1_TEST_HOLE is not None:
    phase = "1"
    hole = P1_TEST_HOLE
elif P2_TEST_HOLE is not None:
    phase = "2"
    hole = P2_TEST_HOLE
else:
    ValueError("Test hole specified incorrectly?")

plot_title = f"Test hole {hole} from phase {phase}, MAE={mae_best_trn:.2f}"
plt.figure(figsize=(6, 12), dpi=100)
plt.title(plot_title)
plt.plot(pred_best_trn, depth, c="red", alpha=0.7, label="Predictions (best trn. model)")
plt.plot(ground_truth, depth, c="black", label="Ground truth")
plt.hlines(
    y=[test_depth],
    xmin=ground_truth.min(),
    xmax=ground_truth.max(),
    colors="gray",
    linestyles="--",
    label="Test start",
)
plt.ylabel("Depth [ft]")
plt.xlabel("BHCT [F]")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig(f"{model_dir}/bhct_test_well_{hole}_p{phase}.png")
writer.add_figure(f"{model_name}, test_well={hole} phase={phase}", plt.gcf())
plt.close()

# Close the TensorBoard writer
writer.close()
