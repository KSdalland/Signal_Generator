import numpy as np


def white_noise(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    return amplitude * rng.standard_normal(len(t))


def pink_noise(t: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    n = len(t)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero at DC
    power = 1.0 / np.sqrt(freqs)
    power[0] = 0.0  # zero DC component
    spectrum = np.fft.rfft(white) * power
    pink = np.fft.irfft(spectrum, n=n)
    pink = pink / (pink.std() + 1e-10)
    return amplitude * pink


def drift(t: np.ndarray, amplitude: float, freq: float, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0, 2 * np.pi)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)
