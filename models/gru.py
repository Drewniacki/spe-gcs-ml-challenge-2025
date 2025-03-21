import torch
import torch.nn as nn

from config import DROPOUT


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, use_state):
        super(GRU, self).__init__()
        output_size = 1
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias=True,
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=False,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)  # LayerNorm after GRU
        self.fc = nn.Linear(hidden_size, output_size)
        self.seq_len = seq_len
        self.use_state = use_state

    def forward(self, x, h):
        """Input shape [batch, steps, features]."""
        if not self.use_state:
            h = None
        x, h = self.gru(x, h)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x, h

    def process_sequence(self, x):
        """Process an entire sequence at once.

        The input shape must be [num_steps, num_features].
        """
        assert x.dim() == 2, f"Incorrect input shape: {x.size()}"
        num_items, _ = x.size()

        device = x.get_device()
        if device == -1:
            device = "cpu"
        y = torch.zeros((num_items, 1), device=device)

        if self.use_state:
            i = 0
            state = None
            while i < num_items:
                y_next, state = self.forward(x[i : i + self.seq_len, :], state)
                y[i : i + self.seq_len] = y_next
                i += self.seq_len
        else:
            i = 0
            state = None
            while i < num_items:
                state = None
                y_step, state = self.forward(x[i : i + self.seq_len, :], state)

                if i == 0:
                    y[0 : self.seq_len] = y_step
                else:
                    y[i + self.seq_len : i + self.seq_len + 1] = y_step[-1:]
                i += 1

        return y, state
