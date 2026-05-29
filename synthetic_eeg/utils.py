import numpy as np


def seed_all(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_time_axis(duration: float, fs: float) -> np.ndarray:
    return np.arange(0, duration, 1.0 / fs)
