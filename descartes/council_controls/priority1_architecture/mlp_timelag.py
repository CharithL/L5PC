"""
mlp_timelag.py

Phase 1A: Feedforward MLP with Time-Lagged Windows

NON-OSCILLATORY control architecture for DESCARTES pipeline.

Key properties distinguishing this from LSTM:
- NO recurrence (no hidden state carried between timesteps)
- NO gating mechanisms (no forget/input/output gates)
- Model sees a fixed window of past input at each timestep
- Cannot develop oscillatory filtering as an architectural property

Window sizes to test: 50ms, 100ms, 200ms, 500ms (at circuit dt)
"""

import numpy as np
import torch
import torch.nn as nn


class MLPTimeLag(nn.Module):
    def __init__(self, n_input, n_output, window_bins,
                 hidden_sizes=(512, 256, 128), dropout=0.3):
        """
        n_input: number of input neurons
        n_output: number of output neurons
        window_bins: number of past timesteps to include
        hidden_sizes: list of hidden layer sizes
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.window_bins = window_bins
        self.hidden_sizes = hidden_sizes

        input_dim = n_input * window_bins
        layers = []
        prev_dim = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hs
        layers.append(nn.Linear(prev_dim, n_output))
        self.network = nn.Sequential(*layers)

        self.penultimate_size = hidden_sizes[-1] if hidden_sizes else input_dim

    def forward(self, x_window):
        """
        x_window: (batch, n_input * window_bins)
        Returns: (batch, n_output) spike probabilities
        """
        return self.network(x_window)

    def extract_hidden_at(self, x_window):
        """Extract penultimate layer activations for probing."""
        x = x_window
        for layer in list(self.network)[:-1]:
            x = layer(x)
        return x

    def extract_hidden_states(self, x_sequence):
        """Given full input sequence (T, n_input), slide window and extract
        penultimate layer activations at each timestep.

        Returns: (T - window_bins + 1, penultimate_size)
        """
        T, n_in = x_sequence.shape
        if T < self.window_bins:
            return np.zeros((0, self.penultimate_size))

        windows = []
        for t in range(self.window_bins - 1, T):
            start = t - self.window_bins + 1
            window = x_sequence[start:t + 1].flatten()
            windows.append(window)

        windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32)

        with torch.no_grad():
            if windows_tensor.device != next(self.parameters()).device:
                windows_tensor = windows_tensor.to(next(self.parameters()).device)
            hidden = self.extract_hidden_at(windows_tensor)
        return hidden.cpu().numpy()


def create_windowed_dataset(X_sequence, Y_sequence, window_bins):
    """Convert sequential data to windowed format for MLP training."""
    T, n_input = X_sequence.shape
    n_output = Y_sequence.shape[1]

    n_samples = T - window_bins + 1
    if n_samples < 1:
        return np.zeros((0, n_input * window_bins)), np.zeros((0, n_output))

    X_windowed = np.zeros((n_samples, n_input * window_bins), dtype=np.float32)
    for i in range(n_samples):
        X_windowed[i] = X_sequence[i:i + window_bins].flatten()

    Y_aligned = Y_sequence[window_bins - 1:].astype(np.float32)
    return X_windowed, Y_aligned


def train_mlp_timelag(model, X_windowed, Y_aligned, train_mask, test_mask,
                      max_epochs=300, patience=20, lr=1e-3, batch_size=256,
                      device='cpu'):
    """Train MLP with time-lagged windows. Cosine annealing + early stopping."""
    X_train = torch.tensor(X_windowed[train_mask], dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_aligned[train_mask], dtype=torch.float32, device=device)
    X_test = torch.tensor(X_windowed[test_mask], dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_aligned[test_mask], dtype=torch.float32, device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    n_train = len(X_train)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            pred = model(X_train[batch_idx])
            loss = criterion(pred, Y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval_flag = True
        with torch.no_grad():
            val_loss = criterion(model(X_test), Y_test).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred = model(X_test).cpu().numpy().ravel()
        true = Y_test.cpu().numpy().ravel()

    cc = 0.0
    if np.std(pred) > 1e-10 and np.std(true) > 1e-10:
        cc = float(np.corrcoef(pred, true)[0, 1])

    return model, cc, epoch + 1
