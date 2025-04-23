"""
Module: data_ingestion.py

Provides functions to load, merge, and save hydraulic system sensor data
and health profiles, based on predefined sensor groups.
"""
import os
import pandas as pd

# Define sensor groups: filename patterns and sample counts per cycle
SENSOR_GROUPS = {
    "PS":   {"files": [f"PS{i}.txt" for i in range(1, 7)],        "pts": 6000},
    "EPS1": {"files": ["EPS1.txt"],                               "pts": 6000},
    "FS":   {"files": ["FS1.txt", "FS2.txt"],                  "pts": 600},
    "LOW":  {"files": ["TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt",
                         "VS1.txt", "CE.txt", "CP.txt", "SE.txt"], "pts": 60}
}


def load_sensor_data(raw_dir: str) -> pd.DataFrame:
    """
    Load and merge all raw sensor files defined in SENSOR_GROUPS.

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing sensor .txt files.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_cycles, total_sensor_samples) with
        columns named <sensor>_<t>.
    """
    df_list = []

    for group in SENSOR_GROUPS.values():
        for filename in group["files"]:
            filepath = os.path.join(raw_dir, filename)
            df = pd.read_csv(filepath, sep='\t', header=None)
            base = os.path.splitext(filename)[0]
            n_pts = group['pts']
            if df.shape[1] != n_pts:
                raise ValueError(f"Unexpected sample count in {filename}: "
                                 f"got {df.shape[1]}, expected {n_pts}")
            df.columns = [f"{base}_{i}" for i in range(n_pts)]
            df_list.append(df)

    # Concatenate all sensor DataFrames side by side
    return pd.concat(df_list, axis=1)


def load_profile_data(profile_path: str) -> pd.DataFrame:
    """
    Load health profile annotations.

    Parameters
    ----------
    profile_path : str
        Path to 'profile.txt' file with columns:
        cooler_pct, valve_pct, pump_leak, acc_pressure, stable_flag.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_cycles, 5) with named columns.
    """
    col_names = [
        'cooler_pct',
        'valve_pct',
        'pump_leak',
        'acc_pressure',
        'stable_flag'
    ]
    df = pd.read_csv(profile_path, sep='\t', header=None, names=col_names)
    return df


def merge_data(sensor_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate sensor and profile DataFrames side by side.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Sensor readings DataFrame.
    profile_df : pd.DataFrame
        Health profile DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with sensor columns first, then profiles.
    """
    return pd.concat([sensor_df, profile_df], axis=1)


def save_cycles(df: pd.DataFrame, output_path: str) -> None:
    """
    Save merged cycles DataFrame to CSV, creating directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame to save.
    output_path : str
        Path of the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, '..', 'data', 'raw')
    profile_path = os.path.join(raw_dir, 'profile.txt')
    output_path = os.path.join(base_dir, '..', 'data', 'processed', 'hydraulic_cycles.csv')

    print("Loading sensor data...")
    sensors = load_sensor_data(raw_dir)
    print(f"Sensor data shape: {sensors.shape}")

    print("Loading profile data...")
    profile = load_profile_data(profile_path)
    print(f"Profile data shape: {profile.shape}")

    print("Merging datasets...")
    df = merge_data(sensors, profile)
    print(f"Merged data shape: {df.shape}")

    print("Saving to CSV...")
    save_cycles(df, output_path)
    print("Done.")
