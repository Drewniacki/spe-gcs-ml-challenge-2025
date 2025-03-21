# PARAMETERS WHICH CAN BE MODIFIED
# =======================================================================================

# Training parameters
NUM_EPOCHS = 200

# Training and test wells
P1_TRAIN_HOLE: int | list[int] | None = 0  # Options: [0, 1, 2], 0, 1, 2, None
P1_TEST_HOLE: int | None = 0  # Options: 0, 1, 2, None

P2_TRAIN_HOLE: int | list[int] | None = None  # Options: [0, 1], 0, 1, None
P2_TEST_HOLE: int | None = None  # Options: 0, 1, None

TEST_SIZE: int = 16  # Number of last rows to test the model on

# Dataset parameters
SEQ_LEN = 16
BATCH_SIZE = 128

# Model hyperparameters
HIDDEN_SIZE = 16
LR = 2e-3  # Learning rate

# Sanity checks
assert P1_TEST_HOLE is None or P2_TEST_HOLE is None, "Only one test hole is allowed!"
assert P1_TEST_HOLE is not None or P2_TEST_HOLE is not None, "Exactly one test hole must be chosen!"
if P1_TEST_HOLE is not None:
    assert P1_TEST_HOLE == P1_TRAIN_HOLE or P1_TEST_HOLE in P1_TRAIN_HOLE
if P2_TEST_HOLE is not None:
    assert P2_TEST_HOLE == P2_TRAIN_HOLE or P2_TEST_HOLE in P2_TRAIN_HOLE
if isinstance(P1_TRAIN_HOLE, list) and len(P1_TRAIN_HOLE) != 3:
    raise ValueError("Training on 2 wells in Phase 1 is not supported. Train on 1 or all 3.")
assert TEST_SIZE > 0, "At least 1 row must be used for test/validation, sorry:)"

# DO NOT TOUCH
# =======================================================================================
MAX_ABS_X = 0.01  # X noise amplitude (added to normalized input)
MAX_ABS_Y = 0.0  # Y noise amplitude (added to raw target in Fahrenheit degrees)
NUM_LAYERS = 2          # Number of layers in the model
DROPOUT = 0.2           # Dropout rate used in training
USE_STATE_TRN = False   # If True, batch size should be 1
USE_STATE_EVL = False   # It's risky to have a different setting than in training

# Paths
LOG_DIR = "runs"
CHECKPOINT_DIR = "checkpoints"
TENSORBOARD_DIR = "tensorboard"

# Case name
def to_string(x: int | list[int] | None) -> str:
    """Converts a list of ints or an int to a string. E.g. [0, 1] -> '01'."""
    s = ""
    if x is None:
        return "None"
    elif isinstance(x, int):
        x = [x]
    for v in x:
        s += str(v)
    return s

p1_train_h = to_string(P1_TRAIN_HOLE)
p1_test_h = to_string(P1_TEST_HOLE)
p2_train_h = to_string(P2_TRAIN_HOLE)
p2_test_h = to_string(P2_TEST_HOLE)

NAME = f"p1trn{p1_train_h}_p1tst{p1_test_h}_p2trn{p2_train_h}_p2tst{p2_test_h}"

SUP_NAME = (
    f"seq{SEQ_LEN}-"
    f"hid{HIDDEN_SIZE}-"
    f"bch{BATCH_SIZE}-"
    f"test{TEST_SIZE}-"
    f"ep{NUM_EPOCHS}"
)
