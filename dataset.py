import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemperatureDataset(Dataset):
    """PyTorch's Dataset subclass to be used in training with DataLoader."""
    def __init__(self, x_paths: list[str], y_paths: list[str], seq_len: int):

        assert len(x_paths) == len(y_paths), "Different number of CSV files for X and y"
        self.seq_len = seq_len

        # Data is held in lists of numpy arrays
        self.x_arrs = []  # Features
        self.y_arrs = []  # Target

        for xp, yp in zip(x_paths, y_paths):
            self.x_arrs.append(pd.read_csv(xp).to_numpy())
            self.y_arrs.append(pd.read_csv(yp).to_numpy())

        # List with number of rows in each csv
        self.boundaries = []
        for i, x in enumerate(self.x_arrs):
            length = len(x) - self.seq_len
            if i == 0:
                self.boundaries.append(length)
            else:
                self.boundaries.append(self.boundaries[-1] + length)

    def get_index_per_hole(self, exclude: list[int]) -> list[list[int]]:
        """Returns indices for each hole, but excluding test indices.

        The return value is a list of lists of indices.
        Each sublist contains indices for each hole.

        Args:
            exclude: list of indices to exclude, used for test_index

        Returns:
            list of lists of indices for each hole
        """
        holes = []
        for hole_num in range(len(self.boundaries)):
            next_hole_ix = self.boundaries[hole_num]
            prev_hole_ix = 0
            if hole_num > 0:
                prev_hole_ix = self.boundaries[hole_num - 1]
            holes.append([i for i in range(prev_hole_ix, next_hole_ix) if i not in exclude])

        return holes

    def __len__(self):
        """Returns the length of this dataset (number of training sequences)."""
        return self.boundaries[-1]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Returns a sequence based on index. The output values are tensors.

        This method first determines which input CSV should be used.
        Then retrieves a sequence from that CSV.

        The sequences are returned without any padding.

        Output shape: x (seq_len, num_feat), y (seq_len, 1)
        """
        # Find out which hole holds the requested sequence
        i = 0
        while idx > self.boundaries[i]:
            i += 1
            if i > len(self.x_arrs) or i > len(self.y_arrs):
                raise RuntimeError("Incorrect index")

        # Determine if this index marks the beginning of a hole.
        # It can be used to reset state of the GRU/LSTM model.
        hole_start = False
        if idx == self.boundaries[i]:
            hole_start = True

        # Select data from the selected hole
        x = self.x_arrs[i]
        y = self.y_arrs[i]

        # Offset index
        if i > 0:
            idx = idx - self.boundaries[i - 1]

        # Prepare output tuple
        item = (
            torch.tensor(x[idx : idx + self.seq_len], dtype=torch.float32),
            torch.tensor(y[idx : idx + self.seq_len], dtype=torch.float32),
            hole_start,
        )

        # NOTE: Last items may be shorter than seq_len!
        if len(item[0]) < self.seq_len:
            print("WARNING: Returned sequence is shorter than seq_len!")
            print("         It may happen in the case of last items in the dataset")
        return item

    def get_selected(self, index: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected features and targets (x, y) based on a list of indices.

        This method can be used e.g. to extract a contiguous test set,
        to test the model on a full sequence instead on a series of short sequences.

        Args:
            index: indices of items to retrieve from the dataset

        Returns:
            tuple of arrays, x.shape = (num_items, num_feat), y.shape = (num_items, 1)
        """
        num_features = self.x_arrs[0].shape[1]
        x = torch.zeros((len(index), num_features))
        y = torch.zeros((len(index), 1))
        for i, ix in enumerate(index):
            # Returned item is a tuple of sequences
            item = self.__getitem__(ix)
            # I just want to take the first value from each sequence
            x[i], y[i] = item[0][0, :], item[1][0, :]

        # Sanity checks
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        return x, y

    def get_train_test_split(
        self,
        test_hole_num: int,
        test_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns train and test indices based on which hole should be used for test."""

        assert test_hole_num < len(self.boundaries)

        # Test index consists of the last items of the selected hole
        next_hole_ix = self.boundaries[test_hole_num]
        test_index = np.arange(next_hole_ix - test_size, next_hole_ix)

        # Extended test index contains entire test hole
        prev_hole_ix = 0
        if test_hole_num > 0:
            prev_hole_ix = self.boundaries[test_hole_num - 1]
        extended_test_index = np.arange(prev_hole_ix + 1, next_hole_ix)

        # Train index contains everything else
        train_index = np.array(
            [i for i in range(self.__len__()) if (i < test_index[0] or i > test_index[-1])]
        )

        return train_index, test_index, extended_test_index
