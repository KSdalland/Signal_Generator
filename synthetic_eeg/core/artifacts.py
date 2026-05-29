import numpy as np


def _gaussian_pulse(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def eye_blink(t: np.ndarray, rate: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    duration = t[-1]
    n_blinks = max(1, int(rate * duration))
    signal = np.zeros(len(t))
    centers = rng.uniform(0, duration, n_blinks)
    for center in centers:
        signal += amplitude * _gaussian_pulse(t, center, width=0.1)
    return signal


def heartrate(t: np.ndarray, bpm: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    freq = bpm / 60.0
    phase = rng.uniform(0, 2 * np.pi)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def muscle_burst(t: np.ndarray, rate: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    duration = t[-1]
    n_bursts = max(1, int(rate * duration))
    signal = np.zeros(len(t))
    for _ in range(n_bursts):
        center = rng.uniform(0, duration)
        freq = rng.uniform(30, 100)
        phase = rng.uniform(0, 2 * np.pi)
        envelope = _gaussian_pulse(t, center, width=0.05)
        signal += amplitude * envelope * np.sin(2 * np.pi * freq * t + phase)
    return signal


def dead_channel(t: np.ndarray) -> np.ndarray:
    return np.zeros(len(t))


def bad_channel(t: np.ndarray, amplitude_multiplier: float, rng: np.random.Generator) -> np.ndarray:
    return amplitude_multiplier * rng.standard_normal(len(t))
