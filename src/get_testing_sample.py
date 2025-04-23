"""
Script: get_testing_sample.py

Randomly extracts a specified number of cycles from `hydraulic_cycles.csv`,
writes the sensor feature data (no profile columns) to `testing_sample.csv`,
and writes the corresponding target labels to `testing_profile.csv`.

Usage:
    python src/get_testing_sample.py
"""
import pandas as pd
from pathlib import Path

def extract_testing_sample(
    input_csv: Path,
    sample_csv: Path,
    profile_csv: Path,
    n_samples: int = 10,
    random_state: int = 42
) -> None:
    # Load full cycles with profiles
    df = pd.read_csv(input_csv)

    # Sample n random cycles
    sample = df.sample(n=n_samples, random_state=random_state)

    # Define target columns to extract
    profile_cols = ['cooler_pct', 'valve_pct', 'pump_leak', 'acc_pressure']

    # Split into sensor features and profile labels
    features = sample.drop(columns=profile_cols)
    profiles = sample[profile_cols]

    # Ensure output directory exists
    sample_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    features.to_csv(sample_csv, index=False)
    profiles.to_csv(profile_csv, index=False)

    print(f"Extracted {n_samples} cycles:")
    print(f" • Sensor data ➔ {sample_csv}")
    print(f" • Profile labels ➔ {profile_csv}")

if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent.parent
    INPUT_CSV = BASE_DIR / 'data' / 'processed' / 'hydraulic_cycles.csv'
    SAMPLE_CSV = BASE_DIR / 'data' / 'processed' / 'testing_sample.csv'
    PROFILE_CSV = BASE_DIR / 'data' / 'processed' / 'testing_profile.csv'

    extract_testing_sample(
        input_csv=INPUT_CSV,
        sample_csv=SAMPLE_CSV,
        profile_csv=PROFILE_CSV,
        n_samples=10,
        random_state=42
    )