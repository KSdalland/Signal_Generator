import numpy as np
import pandas as pd


def make_epochs(df: pd.DataFrame, epoch_cfg: dict, fs: float, rng: np.random.Generator) -> list:
    """Slice a continuous DataFrame into a list of {"data": DataFrame, "label": int} dicts."""
    length = epoch_cfg["length"]
    overlap = epoch_cfg.get("overlap", 0.0)
    label_blocks = epoch_cfg.get("labels", [])
    shuffle = epoch_cfg.get("shuffle", True)

    samples_per_epoch = int(length * fs)
    step = int((length - overlap) * fs)

    starts = list(range(0, len(df) - samples_per_epoch + 1, step))

    label_sequence = []
    for block in label_blocks:
        label_sequence.extend([block["class"]] * block["count"])

    if shuffle:
        rng.shuffle(label_sequence)

    epochs = []
    for i, start in enumerate(starts[: len(label_sequence)]):
        end = start + samples_per_epoch
        epoch_data = df.iloc[start:end].reset_index(drop=True)
        epochs.append({"data": epoch_data, "label": label_sequence[i]})

    return epochs


def epochs_to_numpy(epochs: list) -> tuple:
    """Convert epoch list to (X, y) numpy arrays shaped (n_trials, n_time, n_channels)."""
    X = np.stack([e["data"].values for e in epochs])
    y = np.array([e["label"] for e in epochs])
    return X, y
