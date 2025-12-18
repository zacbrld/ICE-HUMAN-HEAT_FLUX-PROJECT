
######### Imports #########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import datetime

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


######### Column names used across files #########

TIME_STR_COL   = "time [UTC-OFS=+0100]"   # original time string column (when present)
TIMESTAMP_COL  = "timestamp [us]"         # UNIX timestamp in microseconds
HF_COUNTS_COL  = "hf_a0 [counts]"         # raw heat flux counts from human heat sensors


######### Mapping of the participant IDs #########

# Maps raw IDs from filenames (A11, B11, ...) to anonymized IDs (F1, M1, ...)
ID_MAP = {
    "A11": "F1",
    "E12": "F2",
    "I13": "F3",
    "O14": "F4",
    "U15": "F5",
    "Y16": "F6",
    "A17": "F7",
    "E18": "F8",
    "I19": "F9",
    "O20": "F10",
    "Y22": "F11",
    "A23": "F12",

    "B11": "M1",
    "C12": "M2",
    "D13": "M3",
    "F14": "M4",
    "H16": "M5",
    "J17": "M6",
    "K18": "M7",
    "L19": "M8",
    "M20": "M9",
    "N21": "M10",
    "P22": "M11",
    "Q23": "M12",
}


######### Pressure Data Extraction Utils #########


# Extract participant ID and day from filename
def get_id_and_day_from_filename(path):
    """
    Expected filename format: <ID>-<DAY>_... .csv
    Example: A11-1_someinfo.csv -> participant_id="A11", day=1
    """
    filename = os.path.basename(path)
    name_no_ext = os.path.splitext(filename)[0]

    parts = name_no_ext.split("_")
    first_section = parts[0]

    try:
        id_str, day_str = first_section.split("-")
    except ValueError:
        raise ValueError(f"Unexpected filename format: {filename}")

    participant_id = id_str
    day = int(day_str)
    return participant_id, day


# Load all CSV files from a folder (male or female)
def load_pressure_folder(folder, sex_label, test_ids):
    """
    Loads all pressure CSVs from a given folder and returns:
      - list of DataFrames for training participants
      - list of DataFrames for test participants

    Adds metadata columns:
      - participant_id, day, room_temp_C
    """
    dfs_train = []
    dfs_test  = []

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"{folder} : {len(csv_files)} files found")

    for csv_path in csv_files:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Extract metadata from filename
        pid, day = get_id_and_day_from_filename(csv_path)

        # Add metadata columns
        df["participant_id"] = pid
        df["day"] = day
        df["room_temp_C"] = 24 if day == 1 else 18

        # Sort by time if available
        if "Time" in df.columns:
            df = df.sort_values("Time")

        # Split into train/test by participant ID
        if pid in test_ids:
            dfs_test.append(df)
        else:
            dfs_train.append(df)

    return dfs_train, dfs_test


def plot_data_counts(train_df, test_df, quantity_name="Heat Flux"):
    """
    Bar plots showing the number of data points per participant and per day
    for both train and test sets. Used for analysis but not present in current notebook.
    """

    # Union of participants across train and test
    all_participants = sorted(
        set(train_df["participant_id"].unique()) |
        set(test_df["participant_id"].unique())
    )

    # Count rows per (participant, day)
    train_counts = (train_df.groupby(["participant_id", "day"]).size().unstack("day").fillna(0))
    test_counts = (test_df.groupby(["participant_id", "day"]).size().unstack("day").fillna(0))

    # Ensure same ordering and same day columns on both sides
    all_days = sorted(set(train_counts.columns) | set(test_counts.columns))
    train_counts = (train_counts.reindex(index=all_participants, columns=all_days, fill_value=0))
    test_counts = (test_counts.reindex(index=all_participants, columns=all_days, fill_value=0))

    # Figure setup
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    bar_width = 0.35
    x = np.arange(len(all_participants))

    # Plot train counts
    ax = axes[0]
    for i, day in enumerate(all_days):
        offset = (i - (len(all_days) - 1) / 2) * bar_width
        label = f"Day {int(day)}" if str(day).isdigit() else str(day)
        ax.bar(x + offset,
               train_counts[day].values,
               width=bar_width,
               label=label)

    ax.set_title(f"TRAIN – {quantity_name} data points")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Number of data points")
    ax.set_xticks(x)
    ax.set_xticklabels(all_participants, rotation=45, ha="right")
    ax.legend()

    # Plot test counts
    ax = axes[1]
    for i, day in enumerate(all_days):
        offset = (i - (len(all_days) - 1) / 2) * bar_width
        label = f"Day {int(day)}" if str(day).isdigit() else str(day)
        ax.bar(x + offset,
               test_counts[day].values,
               width=bar_width,
               label=label)

    ax.set_title(f"TEST – {quantity_name} data points")
    ax.set_xlabel("Participant ID")
    ax.set_xticks(x)
    ax.set_xticklabels(all_participants, rotation=45, ha="right")
    ax.legend()

    # Consistent y-limits across both plots
    max_val = max(train_counts.values.max(), test_counts.values.max())
    for ax in axes:
        ax.set_ylim(0, max_val * 1.05)

    plt.tight_layout()
    return fig, axes


######### Human Heat Sensors Data Extraction Utils #########


def parse_heat_folder_name(folder_path):
    """
    Parses folder name like 'CALERA_..._<ID>-<DAY>' and returns (participant_id, day).
    """
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split("_")
    id_day = parts[-1]
    pid_str, day_str = id_day.split("-")
    participant_id = pid_str
    day = int(day_str)
    return participant_id, day

def parse_sensor_id(file_path):
    """
    Extracts the numeric sensor ID from a filename containing 'C<id>'.
    Example: ...C14... -> 14
    """
    base = os.path.basename(file_path)
    m = re.search(r"C(\d+)", base)
    if not m:
        raise ValueError(f"Could not find sensor ID in filename: {base}")
    return int(m.group(1))

def extract_S0(file_path):
    """
    Reads the header of the file and extracts the calibration value S0.
    Only scans the first ~20 lines.
    """
    pattern = re.compile(r"S0\s*=\s*([0-9]+(?:\.\d*)?)")
    with open(file_path, "r") as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Could not find S0 in header of {file_path}")


def compute_heat_flux(df_raw, S0):
    """
    Converts raw counts to heat flux (W/m²) using the provided S0 calibration value.
    """
    # HF [W/m²] = hf_a0 * 1.953125 / S0 / 1000
    return df_raw[HF_COUNTS_COL] * 1.953125 *(1000.0/ S0)


def load_heat_flux_data(human_hf_dir, test_ids):
    """
    Returns two DataFrames:
        heat_train_df : participants not in test_ids
        heat_test_df  : participants in test_ids

    Output columns:
        participant_id, day, Time (datetime), back_flux, thigh_flux
    """
    TIME_STR_COL   = "time [UTC-OFS=+0100]"
    TIMESTAMP_COL  = "timestamp [us]"
    HF_COUNTS_COL  = "hf_a0 [counts]"

    train_rows = []
    test_rows  = []

    # Human heat folders
    folder_paths = sorted(
        d for d in glob.glob(os.path.join(human_hf_dir, "CALERA_*"))
        if os.path.isdir(d)
    )

    for folder in folder_paths:
        pid, day = parse_heat_folder_name(folder)

        # Expect exactly two sensors per folder
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if len(csv_files) != 2:
            print(f"[WARN] {folder}: expected 2 CSVs, found {len(csv_files)} -> skipped")
            continue

        # Collect (sensor_num, filepath) pairs
        cfg_ids = []
        for f in csv_files:
            try:
                num = parse_sensor_id(f)
            except ValueError as e:
                print(f"[WARN] {e} -> skipped")
                continue
            cfg_ids.append((num, f))

        # Determine mapping between sensor numbers and back/thigh positions
        sensor_nums = {n for (n, _) in cfg_ids}
        if sensor_nums == {9, 14}:
            mapping = {9: "back_flux", 14: "thigh_flux"}
        elif sensor_nums == {25, 30}:
            mapping = {25: "back_flux", 30: "thigh_flux"}
        else:
            print(f"[WARN] {folder}: unexpected sensor combination {sensor_nums} -> skipped")
            continue

        dfs_sensor = {}

        for sensor_num, file_path in cfg_ids:
            try:
                # Header is at line 14 -> header=13 (0-indexed)
                df_raw = pd.read_csv(file_path, header=13)
            except Exception as e:
                print(f"[WARN] Could not read {file_path}: {e}")
                continue

            try:
                S0 = extract_S0(file_path)
            except ValueError as e:
                print(f"[WARN] {e}, skipping file.")
                continue

            # Basic column sanity check
            if TIMESTAMP_COL not in df_raw.columns or HF_COUNTS_COL not in df_raw.columns:
                print(f"[WARN] {file_path}: missing {TIMESTAMP_COL} or {HF_COUNTS_COL} -> skipped")
                continue

            # 1) Build an absolute datetime from the UNIX timestamp (us)
            df_raw["Time"] = pd.to_datetime(df_raw[TIMESTAMP_COL], unit="us")

            # 2) Convert counts to W/m²
            df_raw["HF_Wm2"] = compute_heat_flux(df_raw, S0)

            # 3) Keep only Time + flux and rename to back/thigh
            df = df_raw[["Time", "HF_Wm2"]].copy()
            df.rename(columns={"HF_Wm2": mapping[sensor_num]}, inplace=True)

            dfs_sensor[mapping[sensor_num]] = df

        # Require both signals
        if "back_flux" not in dfs_sensor or "thigh_flux" not in dfs_sensor:
            print(f"[WARN] {folder}: missing one of back/thigh -> skipped")
            continue

        # 4) Merge the two sensors on Time (nearest neighbor)
        df_merged = pd.merge_asof(
            dfs_sensor["back_flux"].sort_values("Time"),
            dfs_sensor["thigh_flux"].sort_values("Time"),
            on="Time",
            direction="nearest"
        )

        df_merged["participant_id"] = pid
        df_merged["day"] = day

        df_final = df_merged[["participant_id", "day", "Time", "back_flux", "thigh_flux"]]

        # Split into train / test
        if pid in test_ids:
            test_rows.append(df_final)
        else:
            train_rows.append(df_final)

    # Concatenate and sort for clean outputs
    if train_rows:
        heat_train_df = pd.concat(train_rows, ignore_index=True)
        heat_train_df = heat_train_df.sort_values(["participant_id", "day", "Time"]).reset_index(drop=True)
    else:
        heat_train_df = pd.DataFrame(columns=["participant_id", "day", "Time", "back_flux", "thigh_flux"])

    if test_rows:
        heat_test_df = pd.concat(test_rows, ignore_index=True)
        heat_test_df = heat_test_df.sort_values(["participant_id", "day", "Time"]).reset_index(drop=True)
    else:
        heat_test_df = pd.DataFrame(columns=["participant_id", "day", "Time", "back_flux", "thigh_flux"])

    return heat_train_df, heat_test_df


def plot_heatflux_data_counts(heat_train):
    """
    Bar plot of the number of heat flux samples per participant and per day (train set).
    """
    counts = (
        heat_train
        .groupby(["participant_id", "day"])
        .size()
        .unstack(fill_value=0)
    )

    print(counts)  # useful sanity check

    participants = counts.index.astype(str)
    days        = counts.columns.tolist()

    x = np.arange(len(participants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    # Assumes at most two days; extend if needed
    if len(days) >= 1:
        ax.bar(x - width/2, counts[days[0]], width, label=f"Day {days[0]}", color="red")
    if len(days) >= 2:
        ax.bar(x + width/2, counts[days[1]], width, label=f"Day {days[1]}", color="blue")

    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Number of data points")
    ax.set_title("Number of heat flux data points per participant and day (train)")
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


############## Human Heat Flux and Pressure Mergeing Utils ##############


def merge_heat_pressure_by_group(pressure_df, heat_df, tolerance_seconds=2):
    """
    Time-align pressure features to human heat flux measurements.

    Strategy:
      - left  = heat_df (keeps every ground-truth heat sample)
      - right = pressure_df (adds nearest pressure features in time)
      - merged separately per (participant_id, day)
      - if no pressure sample is within tolerance, pressure columns remain NaN
    """
    merged_list = []

    for (pid, day), df_h in heat_df.groupby(["participant_id", "day"]):

        # Matching pressure rows for this participant/day
        df_p = pressure_df[(pressure_df["participant_id"] == pid) &
                           (pressure_df["day"] == day)]

        # Drop identifiers from pressure to avoid duplicate columns after merge
        if not df_p.empty:
            df_p = df_p.drop(columns=["participant_id", "day"], errors="ignore")

        # merge_asof requires time-sorted frames
        df_h = df_h.sort_values("Time")
        df_p = df_p.sort_values("Time")

        merged = pd.merge_asof(
            df_h,
            df_p,
            on="Time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
        )

        merged_list.append(merged)

    if not merged_list:
        return pd.DataFrame()

    return pd.concat(merged_list, ignore_index=True)


def apply_id_mapping(df, col="participant_id", mapping=ID_MAP):
    """
    Replaces participant_id using ID_MAP (e.g., A11 -> F1).
    Unknown IDs are left unchanged.
    """
    df = df.copy()
    df[col] = df[col].map(mapping).fillna(df[col])
    return df


############### Chair Heat Flux sensor Data Extraction Utils ###############


# 1) Parse folder name F_7_1 / M_8_2 -> ("F7", 1)
def parse_folder_name(folder_path):
    """
    Parses seat heatflux folder names like:
      - F_7_1 -> participant_id="F7", day=1
      - M_8_2 -> participant_id="M8", day=2
    """
    name = os.path.basename(folder_path)
    m = re.match(r"([FM])_(\d+)_(\d+)$", name)
    if not m:
        raise ValueError(f"Unexpected folder name: {name}")
    sex = m.group(1)
    num = int(m.group(2))
    day = int(m.group(3))
    participant_id = f"{sex}{num}"
    return participant_id, day


# 2) Extract sensor ID from filename
def parse_sensor_id_from_file(path):
    """
    Extract sensor ID from filename.

    Handles both:
      - Seat_Heatflux_07_17022023_A11-1.csv   (sensor at the beginning)
      - Seat_Heatflux_23012023_U15-1_07.csv   (sensor at the end)
    """
    base = os.path.basename(path)

    m1 = re.match(r"Seat_Heatflux_(\d+)_\d+_[A-Z]\d+-\d", base)
    if m1:
        return int(m1.group(1))

    m2 = re.match(r"Seat_Heatflux_\d+_[A-Z]\d+-\d_(\d+)", base)
    if m2:
        return int(m2.group(1))

    raise ValueError(f"Unexpected seat heatflux filename: {base}")


# 3) Robust read for chair heatflux files (tries multiple encodings)
def read_seat_file(file_path, flux_colname):
    """
    Reads a seat heatflux file with robust encoding handling.

    Output columns:
      - Time (datetime)
      - <flux_colname> (numeric)
    """
    encodings_to_try = ["utf-8", "latin1", "cp1252"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                skiprows=2,
                header=None,
                names=["Time", flux_colname, "_trash"],
                encoding=enc,
            )
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    else:
        print(f"[ERROR] Could not read {file_path} with encodings {encodings_to_try}")
        raise last_err

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df[["Time", flux_colname]].dropna(subset=["Time"])
    return df


# 4) Main loader + train/test split
def load_chair_heatflux(chair_hf_root, test_ids_raw, id_map=ID_MAP):
    """
    Returns two DataFrames:
      chair_hf_train, chair_hf_test

    Output columns:
      participant_id, day, Time, chair_back_flux, chair_thigh_flux

    Key logic:
      - participant_id and day come from folder name (F_7_1, M_8_2, ...)
      - only keep days where sensors {7, 8} are available (back/thigh)
      - skip folders that only contain {4, 6, 11} (non-target sensors)
      - test_ids_raw are raw IDs (A11, B11, ...) mapped to F/M IDs
    """
    mapped_test_ids = {id_map.get(pid, pid) for pid in test_ids_raw}

    train_rows = []
    test_rows = []

    folder_pattern = os.path.join(chair_hf_root, "*_*_*")
    folders = sorted(d for d in glob.glob(folder_pattern) if os.path.isdir(d))

    for folder in folders:
        try:
            participant_id, day = parse_folder_name(folder)
        except ValueError as e:
            print(f"[WARN] {e} -> skipped folder")
            continue

        file_pattern = os.path.join(folder, "Seat_Heatflux_*.*")
        files = [f for f in glob.glob(file_pattern)
                 if f.lower().endswith((".csv", ".txt"))]

        if not files:
            print(f"[INFO] {folder}: no Seat_Heatflux files -> skipped")
            continue

        parsed = []
        for f in files:
            try:
                sensor_id = parse_sensor_id_from_file(f)
            except ValueError as e:
                print(f"[WARN] {e} -> skipped file")
                continue
            parsed.append((sensor_id, f))

        if not parsed:
            continue

        sensor_set = {s for (s, _) in parsed}

        # Skip if folder only contains non-target sensors
        if sensor_set == {4, 6, 11} or (
            sensor_set.issuperset({4, 6, 11}) and 7 not in sensor_set and 8 not in sensor_set
        ):
            print(f"[INFO] {folder}: only seat sensors {sensor_set} (no 7/8) -> skipped")
            continue

        # Require sensors 7 and 8 (back/thigh)
        if not ({7, 8} <= sensor_set):
            print(f"[WARN] {folder}: unexpected sensor set {sensor_set} (missing 7 or 8) -> skipped")
            continue

        file_back = [f for (s, f) in parsed if s == 7][0]
        file_thigh = [f for (s, f) in parsed if s == 8][0]

        df_back = read_seat_file(file_back, "chair_back_flux")
        df_thigh = read_seat_file(file_thigh, "chair_thigh_flux")

        # Time-align the two seat sensors
        df_merged = pd.merge_asof(
            df_back.sort_values("Time"),
            df_thigh.sort_values("Time"),
            on="Time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=1),
        )

        df_merged["participant_id"] = participant_id
        df_merged["day"] = day

        df_final = df_merged[["participant_id", "day", "Time",
                              "chair_back_flux", "chair_thigh_flux"]]

        if participant_id in mapped_test_ids:
            test_rows.append(df_final)
        else:
            train_rows.append(df_final)

    if train_rows:
        chair_hf_train = (
            pd.concat(train_rows, ignore_index=True)
            .sort_values(["participant_id", "day", "Time"])
            .reset_index(drop=True)
        )
    else:
        chair_hf_train = pd.DataFrame(
            columns=["participant_id", "day", "Time", "chair_back_flux", "chair_thigh_flux"]
        )

    if test_rows:
        chair_hf_test = (
            pd.concat(test_rows, ignore_index=True)
            .sort_values(["participant_id", "day", "Time"])
            .reset_index(drop=True)
        )
    else:
        chair_hf_test = pd.DataFrame(
            columns=["participant_id", "day", "Time", "chair_back_flux", "chair_thigh_flux"]
        )

    return chair_hf_train, chair_hf_test


################### Merging Utils ###################


def merge_chair_with_human_pressure(chair_df, hp_df, tolerance_seconds=2):
    """
    Time-align chair heat flux samples with (human heat flux + pressure) samples.

    Strategy:
      - left  = chair_df (keep every chair sample)
      - right = hp_df    (attach nearest hp row in time)
      - merged separately per (participant_id, day)
    """
    merged_list = []

    for (pid, day), df_chair in chair_df.groupby(["participant_id", "day"]):
        df_hp = hp_df[(hp_df["participant_id"] == pid) & (hp_df["day"] == day)]
        if df_hp.empty:
            continue

        df_chair = df_chair.sort_values("Time").copy()
        df_hp = df_hp.sort_values("Time").copy()

        # Drop identifiers from right side to avoid duplicated columns
        df_hp = df_hp.drop(columns=["participant_id", "day"], errors="ignore")

        merged = pd.merge_asof(
            df_chair,
            df_hp,
            on="Time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
        )

        merged_list.append(merged)

    if not merged_list:
        return pd.DataFrame()

    out = (
        pd.concat(merged_list, ignore_index=True)
        .sort_values(["participant_id", "day", "Time"])
        .reset_index(drop=True)
    )
    return out


#################### Choosing active moments during experiments ####################


def load_activity_intervals(activity_path):
    """
    Reads 'Actual timing of activities per person.xlsx' and returns a DataFrame with:
        participant_id (F1, M3, ...),
        day            (1, 2),
        start_time     (datetime.time),
        end_time       (datetime.time)

    Each participant/day can contain up to 4 sitting sessions.
    """
    raw = pd.read_excel(activity_path, header=None)

    # Row index 1 holds "New ID" columns like F1-1, F1-2, ...
    new_ids = raw.iloc[1, 1:]

    id_pattern = re.compile(r"^([FM])(\d+)-(\d+)$")
    intervals = []

    for col_idx, new_id in new_ids.items():
        if pd.isna(new_id):
            continue

        new_id_str = str(new_id).strip()
        m = id_pattern.match(new_id_str)
        if not m:
            print(f"[WARN] ID inattendu en col {col_idx}: {new_id_str}")
            continue

        participant_id = f"{m.group(1)}{int(m.group(2))}"
        day = int(m.group(3))

        # 4 sittings: rows 3..10 (start/end pairs)
        for sit in range(4):
            row_start = 3 + 2 * sit
            row_end   = 4 + 2 * sit

            if row_end >= raw.shape[0]:
                continue

            start_val = raw.iloc[row_start, col_idx]
            end_val   = raw.iloc[row_end,   col_idx]

            if pd.isna(start_val) or pd.isna(end_val):
                continue

            # Robust parsing for time values
            if isinstance(start_val, datetime.time):
                t_start = start_val
            else:
                t_start = pd.to_datetime(start_val).time()

            if isinstance(end_val, datetime.time):
                t_end = end_val
            else:
                t_end = pd.to_datetime(end_val).time()

            intervals.append({
                "participant_id": participant_id,
                "day": day,
                "start_time": t_start,
                "end_time": t_end,
            })

    intervals_df = pd.DataFrame(intervals)
    intervals_df["Session"] = intervals_df.groupby(["participant_id", "day"]).cumcount() + 1
    intervals_df = intervals_df.sort_values(['participant_id', 'day', 'Session', 'start_time'])
    return intervals_df

def drop_non_finite_rows_numeric(X):
    """
    Removes rows where any numeric column contains NaN, +inf or -inf.
    Non-numeric columns are ignored in the finiteness check.
    """
    X = X.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        print("No numeric columns found, nothing removed.")
        return X, 0

    numeric_values = X[num_cols].to_numpy()
    mask_finite = np.all(np.isfinite(numeric_values), axis=1)

    X_clean = X[mask_finite].copy()
    removed_count = len(X) - len(X_clean)

    print(f"Removed {removed_count} rows with NaN / +inf / -inf in numeric columns.")

    return X_clean, removed_count


def filter_to_active_periods(df, intervals_df):
    """
    Assigns a Session label based on time-of-day activity intervals:
      - Session = 0 : outside any active interval
      - Session = 1..4 : inside the corresponding interval for (participant_id, day)
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["time_of_day"] = df["Time"].dt.time

    df["Session"] = 0

    for (pid, day), grp in df.groupby(["participant_id", "day"]):
        sub_int = intervals_df[
            (intervals_df["participant_id"] == pid) &
            (intervals_df["day"] == day)
        ]
        if sub_int.empty:
            continue

        sub_int = sub_int.sort_values("start_time")

        for i, (_, row) in enumerate(sub_int.iterrows(), start=1):
            if i > 4:
                break

            s = row["start_time"]
            e = row["end_time"]

            mask_session = (grp["time_of_day"] >= s) & (grp["time_of_day"] <= e)
            df.loc[grp.index[mask_session], "Session"] = i

    return df.drop(columns=["time_of_day"])


#################### Saving Cleaned Datasets ####################


def save_cleaned_datasets(df_train, df_test, name):
    """
    Saves cleaned train/test datasets into 'Cleaned_Datasets/' as CSV.
    """
    output_dir = "Cleaned_Datasets/"

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"Train_{name}.csv")
    test_path  = os.path.join(output_dir, f"Test_{name}.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Saved:")
    print(" -", train_path)
    print(" -", test_path)


#################### Smoothing the datas ####################


# 1) Smoothing function (Savitzky–Golay) by participant/day groups
def smooth_signals(df,
                   signal_cols,
                   sg_window=51,
                   sg_poly=3,
                   group_cols=None):
    """
    For each column in `signal_cols`, adds a new column:
        col + "_sg" : Savitzky–Golay smoothed version of `col`.

    Smoothing is applied independently within each group defined by `group_cols`
    (typically ['participant_id', 'day']). If group_cols is None, the entire
    DataFrame is treated as one group.
    """
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    # Sort to guarantee correct temporal order before smoothing
    if group_cols is not None:
        sort_cols = group_cols + ["Time"]
    else:
        sort_cols = ["Time"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Create placeholder columns for smoothed signals
    for col in signal_cols:
        df[col + "_sg"] = np.nan

    # Build the grouping iterator
    if group_cols is not None:
        grouped = df.groupby(group_cols, group_keys=False)
    else:
        grouped = [(None, df)]

    # Apply Savitzky–Golay per group and per signal
    for _, g in grouped:
        idx = g.index
        for col in signal_cols:
            series = g[col]

            if len(series) >= sg_window:
                df.loc[idx, col + "_sg"] = savgol_filter(
                    series.to_numpy(),
                    sg_window,
                    sg_poly,
                    mode="interp",
                )
            else:
                # If the group is too short, keep original values
                df.loc[idx, col + "_sg"] = series.to_numpy()

    return df


# 2) Helper to: (1) smooth, (2) overwrite original cols, (3) drop _sg
def make_smoothed_copy(df, signal_cols, sg_window=51, sg_poly=3, group_cols=None):
    """
    Returns a DataFrame where:
      - each column in `signal_cols` is replaced by its smoothed version,
      - helper *_sg columns are removed,
      - all other columns are preserved.
    """
    original_cols = df.columns.tolist()

    df_s = smooth_signals(
        df,
        signal_cols=signal_cols,
        sg_window=sg_window,
        sg_poly=sg_poly,
        group_cols=group_cols,
    )

    # Overwrite original signals with their smoothed version
    for col in signal_cols:
        sg_col = col + "_sg"
        if sg_col in df_s.columns:
            df_s[col] = df_s[sg_col]

    # Drop temporary columns
    sg_cols = [col + "_sg" for col in signal_cols if col + "_sg" in df_s.columns]
    df_s = df_s.drop(columns=sg_cols)

    # Restore original ordering (excluding helper cols)
    df_s = df_s[[c for c in original_cols if c in df_s.columns]]

    return df_s


def plot_raw_vs_smoothed(
    raw_df,
    smooth_df,
    time_col,
    signal_cols,
    start_time,
    end_time
):
    """
    Plots raw vs smoothed signals over a given time window.
    """
    mask_raw = (raw_df[time_col] >= start_time) & (raw_df[time_col] <= end_time)
    mask_smooth = (smooth_df[time_col] >= start_time) & (smooth_df[time_col] <= end_time)

    raw_win = raw_df[mask_raw].copy()
    smooth_win = smooth_df[mask_smooth].copy()

    if raw_win.empty or smooth_win.empty:
        raise ValueError("No samples found in the selected window.")

    print(f"Raw samples: {len(raw_win)}, Smoothed samples: {len(smooth_win)}")

    for signal in signal_cols:

        # Skip signals not present in either DataFrame
        if signal not in raw_win.columns or signal not in smooth_win.columns:
            continue

        plt.figure(figsize=(14, 4))

        # Raw series
        plt.plot(raw_win["Time"], raw_win[signal],
                 alpha=0.35, label=f"{signal} (raw)")

        # Smoothed series
        plt.plot(smooth_win["Time"], smooth_win[signal],
                 linewidth=2, label=f"{signal} (smoothed)")

        plt.title(f"{signal}: Raw vs Smoothed\n{start_time} → {end_time}")
        plt.xlabel("Time")
        plt.ylabel(signal)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


#################### Feature engineering ####################


def fill_lag_with_first(series):
    """
    For lag features: fill initial NaNs with the first valid value of the session/series.
    """
    idx = series.first_valid_index()
    first = series.loc[idx]
    return series.fillna(first)


def fill_roll_mean_with_past(series):
    """
    For rolling mean features: if a value is NaN, replace it with the mean of past observed values.
    If no past exists yet, fallback to the first valid value of the series.
    """
    values = []
    out = []

    for v in series:
        if pd.isna(v):
            if len(values) == 0:
                idx = series.first_valid_index()
                first = series.loc[idx]
                out.append(first)
            else:
                out.append(np.mean(values))
        else:
            values.append(v)
            out.append(v)

    return pd.Series(out, index=series.index)


def fill_roll_std_with_past(series):
    """
    For rolling std features: if a value is NaN, replace it with the std of past observed values.
    If no past exists yet, use 0.0 (no variation observed so far).
    """
    values = []
    out = []

    for v in series:
        if pd.isna(v):
            if len(values) == 0:
                out.append(0.0)
            else:
                out.append(float(np.std(values, ddof=0)))
        else:
            values.append(v)
            out.append(v)

    return pd.Series(out, index=series.index)


def impute_historical_features(df):
    """
    Imputes missing values for engineered historical features using session-wise logic:
      - *_lag      -> filled with first valid value in the session
      - *_roll_*mean -> filled with mean of past values
      - *_roll_*std  -> filled with std of past values (or 0 if no past)
    """
    for col in df.columns:
        if "_lag" in col:
            df[col] = (
                df.groupby("Session")[col]
                  .transform(fill_lag_with_first)
            )
        elif "_roll" in col and "_mean" in col:
            df[col] = (
                df.groupby("Session")[col]
                  .transform(fill_roll_mean_with_past)
            )
        elif "_roll" in col and "_std" in col:
            df[col] = (
                df.groupby("Session")[col]
                  .transform(fill_roll_std_with_past)
            )

    return df

