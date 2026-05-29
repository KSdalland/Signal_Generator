import numpy as np


def generate_sinusoid(
    freq: float,
    amplitude: float,
    phase: float,
    t: np.ndarray,
    start: float = None,
    stop: float = None,
) -> np.ndarray:
    wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
    if start is not None or stop is not None:
        mask = np.ones(len(t), dtype=bool)
        if start is not None:
            mask &= t >= start
        if stop is not None:
            mask &= t <= stop
        wave = wave * mask
    return wave


def generate_channel_signal(components: list, t: np.ndarray) -> np.ndarray:
    signal = np.zeros(len(t))
    for comp in components:
        signal += generate_sinusoid(
            comp["freq"],
            comp["amplitude"],
            comp.get("phase", 0.0),
            t,
            start=comp.get("start"),
            stop=comp.get("stop"),
        )
    return signal
