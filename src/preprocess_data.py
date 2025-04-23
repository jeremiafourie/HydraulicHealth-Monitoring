"""
Module: feature_engineering.py

Provides modular functions to compute per-cycle summary statistics (mean, std, slope),
FFT-based features, and assemble a features table for hydraulic system monitoring.
"""
import os
import numpy as np
import pandas as pd

# Definitions of sensor prefixes and their sample counts per cycle
SENSOR_DEFS = [
    (f"PS{i}", 6000) for i in range(1, 7)
] + [
    ("EPS1", 6000)
] + [
    ("FS1", 600), ("FS2", 600)
] + [
    ("TS1", 60), ("TS2", 60), ("TS3", 60), ("TS4", 60),
    ("VS1", 60), ("CE", 60), ("CP", 60), ("SE", 60)
]

TARGET_COLUMNS = ["cooler_pct", "valve_pct", "pump_leak", "acc_pressure"]


def load_cycles(path: str) -> pd.DataFrame:
    """
    Load the merged hydraulic cycle DataFrame.
    """
    return pd.read_csv(path)


def summary_stats(df: pd.DataFrame, prefix: str, n_points: int) -> pd.DataFrame:
    """
    Compute mean, std, and slope for a given sensor prefix over each cycle.
    """
    cols = [f"{prefix}_{i}" for i in range(n_points)]
    arr = df[cols].values
    x = np.arange(n_points)
    return pd.DataFrame({
        f"{prefix}_mean":  arr.mean(axis=1),
        f"{prefix}_std":   arr.std(axis=1),
        f"{prefix}_slope": np.polyfit(x, arr.T, 1)[0]
    })


def build_summary_features(cycles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over SENSOR_DEFS and build a DataFrame of all summary stats.
    """
    feat_dfs = []
    for prefix, n in SENSOR_DEFS:
        feat_dfs.append(summary_stats(cycles_df, prefix, n))
    return pd.concat(feat_dfs, axis=1)


def fft_features(df: pd.DataFrame, prefix: str, n_bins: int) -> pd.DataFrame:
    """
    Compute the first n_bins FFT magnitudes for a sensor prefix per cycle.
    """
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    arr = df[cols].values
    # real FFT: returns n_bins+1 values including DC; drop DC term
    mag = np.abs(np.fft.rfft(arr, axis=1))[:, 1 : n_bins + 1]
    col_names = [f"{prefix}_fft_{i}" for i in range(1, n_bins + 1)]
    return pd.DataFrame(mag, columns=col_names)


def build_fft_features(cycles_df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Build FFT features for all sensors defined in SENSOR_DEFS.
    """
    fft_dfs = []
    for prefix, _ in SENSOR_DEFS:
        fft_dfs.append(fft_features(cycles_df, prefix, n_bins))
    return pd.concat(fft_dfs, axis=1)


def assemble_features(cycles_df: pd.DataFrame,
                      include_summary: bool = True,
                      include_fft: bool = False,
                      fft_bins: int = 5) -> pd.DataFrame:
    """
    Assemble the feature table by combining summary stats and optional FFT features,
    then appending target columns.
    """
    parts = []
    if include_summary:
        parts.append(build_summary_features(cycles_df))
    if include_fft:
        parts.append(build_fft_features(cycles_df, fft_bins))

    features_df = pd.concat(parts, axis=1)

    # attach target labels
    targets_df = cycles_df[TARGET_COLUMNS].reset_index(drop=True)
    return pd.concat([features_df, targets_df], axis=1)


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the features DataFrame to CSV, creating directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    cycles_path = os.path.join(base, '..', 'data', 'processed', 'hydraulic_cycles.csv')
    features_path = os.path.join(base, '..', 'data', 'processed', 'features.csv')

    print("Loading cycles...")
    cycles = load_cycles(cycles_path)

    print("Building features...")
    feats = assemble_features(cycles, include_summary=True, include_fft=False)
    print(f"Features shape: {feats.shape}")

    print("Saving features...")
    save_features(feats, features_path)
    print("Done.")
