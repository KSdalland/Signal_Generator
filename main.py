import argparse
import matplotlib.pyplot as plt
from synthetic_eeg import EEGGenerator, plot_continuous, plot_psd, plot_epochs


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic EEG data generator")
    parser.add_argument(
        "--config",
        default="synthetic_eeg/config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--save", action="store_true", help="Save output to file (path set in config)")
    parser.add_argument("--plot", action="store_true", help="Show plots after generation")
    return parser.parse_args()


def main():
    args = parse_args()

    gen = EEGGenerator(args.config)
    fs = gen.cfg["global"]["fs"]

    print("Generating continuous signal...")
    df = gen.generate()
    print(f"  Shape: {df.shape}  ({df.shape[0]} samples x {df.shape[1]} channels)")

    X, y = None, None
    if gen.cfg.get("epochs", {}).get("enabled", False):
        print("Generating epochs...")
        X, y = gen.generate_epochs()
        print(f"  Shape: {X.shape}  (trials x time x channels)")
        label_counts = {int(v): int((y == v).sum()) for v in set(y.tolist())}
        print(f"  Labels: {label_counts}")

    if args.save:
        print("Saving...")
        gen.save(df)
        print(f"  Saved to: {gen.cfg['output']['path']}")

    if args.plot:
        channel_names = list(df.columns)
        plot_continuous(df, title="Synthetic EEG — Continuous")
        plot_psd(df, fs=fs, title="Power Spectral Density")
        if X is not None:
            plot_epochs(X, y, channel_names=channel_names, fs=fs)
        plt.show()


if __name__ == "__main__":
    main()
