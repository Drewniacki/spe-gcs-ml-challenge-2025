import torch

class TorchStandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 0.

    def fit(self, X: torch.Tensor):
        """Compute mean and standard deviation from data."""
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True, unbiased=False)  # Use unbiased=False to match Scikit-learn

    def transform(self, X: torch.Tensor):
        """Apply standardization using stored mean and std."""
        return (X - self.mean) / (self.std + 1e-8)  # Add epsilon for numerical stability

    def fit_transform(self, X: torch.Tensor):
        """Fit to data and transform it."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: torch.Tensor):
        """Revert transformation to original scale."""
        return X * self.std + self.mean

