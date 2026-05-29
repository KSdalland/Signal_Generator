import numpy as np
from .signal import generate_channel_signal
from .noise import white_noise, pink_noise, drift
from .artifacts import eye_blink, heartrate, muscle_burst, dead_channel, bad_channel


def build_channel(channel_cfg: dict, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    artifact_cfg = channel_cfg.get("artifacts", {})

    if artifact_cfg.get("dead_channel", {}).get("enabled", False):
        return dead_channel(t)

    components = channel_cfg.get("signal", {}).get("components", [])
    signal = generate_channel_signal(components, t)

    noise_cfg = channel_cfg.get("noise", {})

    if noise_cfg.get("white", {}).get("enabled", False):
        cfg = noise_cfg["white"]
        signal += white_noise(t, cfg["amplitude"], rng)

    if noise_cfg.get("pink", {}).get("enabled", False):
        cfg = noise_cfg["pink"]
        signal += pink_noise(t, cfg["amplitude"], rng)

    if noise_cfg.get("drift", {}).get("enabled", False):
        cfg = noise_cfg["drift"]
        signal += drift(t, cfg["amplitude"], cfg.get("freq", 0.1), rng)

    if artifact_cfg.get("eye_blink", {}).get("enabled", False):
        cfg = artifact_cfg["eye_blink"]
        signal += eye_blink(t, cfg.get("rate", 0.2), cfg["amplitude"], rng)

    if artifact_cfg.get("heartrate", {}).get("enabled", False):
        cfg = artifact_cfg["heartrate"]
        signal += heartrate(t, cfg.get("bpm", 60), cfg["amplitude"], rng)

    if artifact_cfg.get("muscle_burst", {}).get("enabled", False):
        cfg = artifact_cfg["muscle_burst"]
        signal += muscle_burst(t, cfg.get("rate", 0.05), cfg["amplitude"], rng)

    if artifact_cfg.get("bad_channel", {}).get("enabled", False):
        cfg = artifact_cfg["bad_channel"]
        signal += bad_channel(t, cfg.get("amplitude_multiplier", 5.0), rng)

    return signal
