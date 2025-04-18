import pandas as pd
import glob
import os

# 1. Point to your raw data folder
RAW_DIR = "../data/raw"
PROCESSED_CSV = "../data/processed/hydraulic_cycles.csv"

# 2. Define your sensor files and their sampling rates
#    (matches the README: PS1–PS6 + EPS1 @100Hz, FS1–FS2 @10Hz, TS1–TS4, VS1, CE, CP, SE @1Hz)
sensor_groups = {
    "PS":    {"files": [f"PS{i}.txt"    for i in range(1,7)],   "points": 6000},
    "EPS1":  {"files": ["EPS1.txt"],        "points": 6000},
    "FS":    {"files": ["FS1.txt","FS2.txt"],"points": 600},
    "LOW":   {"files": ["TS1.txt","TS2.txt","TS3.txt","TS4.txt",
                        "VS1.txt","CE.txt","CP.txt","SE.txt"], "points": 60},
}

# 3. Load each sensor file into a DataFrame, name its columns, and collect into a list
sensor_dfs = []
for group in sensor_groups.values():
    for fname in group["files"]:
        path = os.path.join(RAW_DIR, fname)
        # read tab‑delimited, no header
        df = pd.read_csv(path, sep="\t", header=None)
        # generate column names like 'PS1_0'...'PS1_5999'
        base = os.path.splitext(fname)[0]  # e.g. 'PS1'
        df.columns = [f"{base}_{i}" for i in range(group["points"])]
        sensor_dfs.append(df)

# 4. Concatenate all sensor DataFrames side-by-side
#    They all have 2205 rows (one per 60‑s cycle), so axis=1 works
sensors = pd.concat(sensor_dfs, axis=1)

# 5. Load the target conditions (profile.txt)
profile_cols = [
    "cooler_pct",   # 3..100
    "valve_pct",    # 73..100
    "pump_leak",    # 0..2
    "acc_pressure", # 90..130
    "stable_flag"   # 0 or 1
]
profile = pd.read_csv(
    os.path.join(RAW_DIR, "profile.txt"),
    sep="\t",
    header=None,
    names=profile_cols
)

# 6. Combine sensors + profile into one DataFrame
df = pd.concat([sensors, profile], axis=1)

# 7. Save to disk for easy reuse
df.to_csv(PROCESSED_CSV, index=False)
print(f"Saved processed data to {PROCESSED_CSV}, shape = {df.shape}")
