import numpy as np
import pandas as pd

from .utils import seed_all, make_time_axis
from .core.channel import build_channel
from .core.epoch import make_epochs, epochs_to_numpy
from .io import load_config, save_csv, save_hdf5


class EEGGenerator:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self._rng = seed_all(self.cfg["global"].get("seed", 42))

    def generate(self, duration: float = None) -> pd.DataFrame:
        fs = self.cfg["global"]["fs"]
        d = duration if duration is not None else self.cfg["global"]["duration"]
        t = make_time_axis(d, fs)
        data = {}
        for ch_cfg in self.cfg["channels"]:
            data[ch_cfg["name"]] = build_channel(ch_cfg, t, self._rng)
        return pd.DataFrame(data, index=t)

    def generate_epochs(self) -> tuple:
        """Return (X, y) numpy arrays shaped (n_trials, n_time, n_channels)."""
        epoch_cfg = self.cfg.get("epochs", {})
        if not epoch_cfg.get("enabled", False):
            raise ValueError("Set epochs.enabled: true in your config to use this method.")

        fs = self.cfg["global"]["fs"]
        length = epoch_cfg["length"]
        overlap = epoch_cfg.get("overlap", 0.0)
        n_epochs = sum(b["count"] for b in epoch_cfg.get("labels", []))
        step = length - overlap
        required_duration = n_epochs * step + length

        configured_duration = self.cfg["global"]["duration"]
        df = self.generate(duration=max(configured_duration, required_duration))

        epochs = make_epochs(df, epoch_cfg, fs, self._rng)
        return epochs_to_numpy(epochs)

    def save(self, df: pd.DataFrame = None):
        output_cfg = self.cfg.get("output", {})
        fmt = output_cfg.get("format", "csv")
        path = output_cfg.get("path", "./output.csv")
        if df is None:
            df = self.generate()
        if fmt == "csv":
            save_csv(df, path)
        elif fmt == "hdf5":
            save_hdf5(df, path)
        else:
            raise ValueError(f"Unknown output format: {fmt}")
