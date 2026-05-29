import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_continuous(df: pd.DataFrame, title: str = "Synthetic EEG", figsize: tuple = None) -> plt.Figure:
    """One subplot per channel, shared time axis."""
    n_channels = len(df.columns)
    figsize = figsize or (14, 2.5 * n_channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        ax.plot(df.index, df[col], linewidth=0.7, color="steelblue")
        ax.set_ylabel(col, rotation=0, labelpad=40, va="center")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_psd(df: pd.DataFrame, fs: float, title: str = "Power Spectral Density", figsize: tuple = None) -> plt.Figure:
    """FFT-based PSD in dB for each channel."""
    n_channels = len(df.columns)
    figsize = figsize or (10, 3 * n_channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        n = len(df)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(df[col].values)) ** 2 / n
        power_db = 10 * np.log10(power + 1e-10)
        ax.plot(freqs, power_db, linewidth=0.8, color="darkorange")
        ax.set_ylabel(f"{col}\n(dB)", va="center")
        ax.set_xlim(0, fs / 2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_epochs(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list = None,
    fs: float = None,
    n_per_class: int = 10,
    figsize: tuple = None,
) -> plt.Figure:
    """
    Overlay individual trials per class with a bold class mean.

    X : (n_trials, n_time, n_channels)
    y : (n_trials,)
    """
    n_trials, n_time, n_channels = X.shape
    channel_names = channel_names or [f"Ch{i}" for i in range(n_channels)]
    classes = sorted(set(y.tolist()))
    colors = plt.cm.tab10.colors

    t = np.arange(n_time) / fs if fs is not None else np.arange(n_time)
    x_label = "Time (s)" if fs is not None else "Sample"

    figsize = figsize or (12, 3 * n_channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, ch_idx in zip(axes, range(n_channels)):
        for cls_idx, cls in enumerate(classes):
            trials = X[y == cls, :, ch_idx][:n_per_class]
            color = colors[cls_idx % len(colors)]
            for trial in trials:
                ax.plot(t, trial, alpha=0.25, linewidth=0.6, color=color)
            ax.plot(t, trials.mean(axis=0), linewidth=2.0, color=color, label=f"Class {cls} mean")
        ax.set_ylabel(channel_names[ch_idx], rotation=0, labelpad=40, va="center")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper right", framealpha=0.7)
    axes[-1].set_xlabel(x_label)
    fig.suptitle("Epochs by Class")
    fig.tight_layout()
    return fig
