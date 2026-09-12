from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Leave this as None to choose the latest race with at least five finishers.
# After checking that the race was dry and had no red flag, you can set it to
# a specific raceId from races.csv for a reproducible submission.
RACE_ID = None
MAX_DRIVERS = 10

def check_required_files():
    required = ["lap_times.csv", "pit_stops.csv", "results.csv", "races.csv"]
    missing = [name for name in required if not (DATA_DIR / name).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required CSV files in the data folder. "
            f"Please add: {', '.join(missing)}"
        )


def load_data():
    lap_times = pd.read_csv(DATA_DIR / "lap_times.csv")
    pit_stops = pd.read_csv(DATA_DIR / "pit_stops.csv")
    results = pd.read_csv(DATA_DIR / "results.csv")
    races = pd.read_csv(DATA_DIR / "races.csv")
    return lap_times, pit_stops, results, races


def clean_lap_times(lap_times):
    raw_count = len(lap_times)

    lap_times = lap_times.copy()
    lap_times = lap_times.dropna(subset=["milliseconds"])
    lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")
    lap_times = lap_times[lap_times["milliseconds"] > 0].copy()

    removed_laps = raw_count - len(lap_times)
    print(f"Removed laps during cleaning: {removed_laps}")

    return lap_times, removed_laps


def pick_race(races, results):
    """Pick one race, then use its first 5-10 classified finishers."""
    print("Finding a suitable race...\n")

    finished = results[results["statusId"] == 1].copy()
    race_summary = finished.groupby("raceId", as_index=False).agg(
        finished_drivers=("driverId", "nunique")
    )
    race_summary = race_summary.merge(races, on="raceId", how="inner")
    race_summary = race_summary[race_summary["finished_drivers"] >= 5]

    if RACE_ID is not None:
        race_summary = race_summary[race_summary["raceId"] == RACE_ID]

    if race_summary.empty:
        raise RuntimeError(
            "No suitable race was found. Set RACE_ID to a race with at least "
            "five finishers and check that it was dry with no red flag."
        )

    # The latest eligible race is convenient for a first run. For a report,
    # set RACE_ID above after verifying the race conditions yourself.
    chosen_race = race_summary.sort_values(["year", "raceId"]).iloc[-1]
    print(f"Chosen race: {chosen_race['year']} - {chosen_race['name']}")
    print(f"Race ID: {chosen_race['raceId']}")
    print(f"Available finishers: {int(chosen_race['finished_drivers'])}")
    print(
        "Reminder: verify this race was dry and had no red flag in the "
        "Kaggle race information before submitting."
    )

    return int(chosen_race["raceId"])


def select_finished_drivers(results, race_id):
    """Keep 5-10 finishers so the comparison stays inside one race."""
    finishers = results[
        (results["raceId"] == race_id) & (results["statusId"] == 1)
    ].copy()
    finishers["position_number"] = pd.to_numeric(
        finishers["position"], errors="coerce"
    )
    finishers = finishers.sort_values(
        ["position_number", "driverId"], na_position="last"
    )
    drivers = finishers["driverId"].drop_duplicates().head(MAX_DRIVERS).tolist()

    if len(drivers) < 5:
        raise RuntimeError("The selected race has fewer than five usable finishers.")

    print(f"Using {len(drivers)} finished drivers: {drivers}")
    return drivers


def build_stint_features(lap_times, pit_stops, race_id, driver_ids):
    lap_times = lap_times[
        (lap_times["raceId"] == race_id)
        & (lap_times["driverId"].isin(driver_ids))
    ].copy()

    if lap_times.empty:
        raise RuntimeError(f"No lap data found for race ID {race_id}")

    pit_stops = pit_stops[pit_stops["raceId"] == race_id].copy()
    pit_stops = pit_stops[["raceId", "driverId", "lap"]].dropna()

    rows = []

    for driver_id, driver_group in lap_times.groupby("driverId"):
        driver_group = driver_group.sort_values("lap").copy()
        stop_laps = sorted(pit_stops[pit_stops["driverId"] == driver_id]["lap"].tolist())

        for _, row in driver_group.iterrows():
            previous_stops = [stop for stop in stop_laps if stop < int(row["lap"])]
            last_stop = previous_stops[-1] if previous_stops else 0

            current_stint = len(previous_stops) + 1
            tire_age = int(row["lap"]) - int(last_stop)

            new_row = row.to_dict()
            new_row["stint_order"] = current_stint
            new_row["tire_age"] = tire_age
            new_row["lap_number_in_stint"] = int(row["lap"]) - last_stop

            rows.append(new_row)

    enriched = pd.DataFrame(rows)
    enriched["lap_number_in_stint"] = enriched.groupby(["driverId", "stint_order"])["lap"].rank(method="first").astype(int)

    return enriched


def split_by_stint(enriched):
    train_frames = []
    test_frames = []

    for driver_id, driver_group in enriched.groupby("driverId"):
        unique_stints = sorted(driver_group["stint_order"].unique())

        if len(unique_stints) < 2:
            train_frames.append(driver_group)
            continue

        split_index = max(1, int(len(unique_stints) * 0.8))
        train_stints = unique_stints[:split_index]
        test_stints = unique_stints[split_index:]

        train_frames.append(driver_group[driver_group["stint_order"].isin(train_stints)])
        test_frames.append(driver_group[driver_group["stint_order"].isin(test_stints)])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    if test_df.empty:
        raise RuntimeError(
            "Not enough stints for a valid stint-based train/test split. "
            "Try another race or use a race with more laps."
        )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    return train_df, test_df


def build_features(df, include_tire_age=False):
    features = df[["lap", "driverId", "stint_order"]].copy()

    if include_tire_age:
        features["tire_age"] = df["tire_age"]
        features["tire_age_sq"] = df["tire_age"] ** 2

    return features


def evaluate_model(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def plot_one_stint(test_df, baseline_model, tire_model):
    plot_driver = test_df["driverId"].mode().iloc[0]
    plot_stint = test_df[test_df["driverId"] == plot_driver]["stint_order"].min()

    plot_df = test_df[
        (test_df["driverId"] == plot_driver) &
        (test_df["stint_order"] == plot_stint)
    ].sort_values("lap").copy()

    if plot_df.empty:
        raise RuntimeError("No test stint data available for plotting.")

    baseline_features = build_features(plot_df, include_tire_age=False)
    tire_features = build_features(plot_df, include_tire_age=True)

    plot_df["baseline_pred"] = baseline_model.predict(baseline_features)
    plot_df["tire_pred"] = tire_model.predict(tire_features)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["lap"], plot_df["milliseconds"], label="Actual", marker="o", linewidth=2)
    plt.plot(plot_df["lap"], plot_df["baseline_pred"], label="Baseline prediction", marker="x", linewidth=2)
    plt.plot(plot_df["lap"], plot_df["tire_pred"], label="Tire-age prediction", marker="s", linewidth=2)
    plt.title(f"Predicted vs Actual Lap Times for Driver {plot_driver} - Stint {plot_stint}")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (milliseconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual_stint.png", dpi=150)
    plt.close()

    print(f"Saved plot to: {OUTPUT_DIR / 'predicted_vs_actual_stint.png'}")


def main():
    check_required_files()

    lap_times, pit_stops, results, races = load_data()
    lap_times, removed_laps = clean_lap_times(lap_times)
    pd.DataFrame(
        {
            "raw_laps": [removed_laps + len(lap_times)],
            "removed_laps": [removed_laps],
            "remaining_laps": [len(lap_times)],
        }
    ).to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False)

    chosen_race_id = pick_race(races, results)
    driver_ids = select_finished_drivers(results, chosen_race_id)

    enriched = build_stint_features(
        lap_times, pit_stops, chosen_race_id, driver_ids
    )
    train_df, test_df = split_by_stint(enriched)

    print("\nTraining baseline model...")
    baseline_X_train = build_features(train_df, include_tire_age=False)
    baseline_X_test = build_features(test_df, include_tire_age=False)
    y_train = train_df["milliseconds"]
    y_test = test_df["milliseconds"]

    baseline_model = LinearRegression()
    baseline_model.fit(baseline_X_train, y_train)
    baseline_pred = baseline_model.predict(baseline_X_test)

    baseline_metrics = evaluate_model(y_test, baseline_pred)

    print("\nTraining tire-age model...")
    tire_X_train = build_features(train_df, include_tire_age=True)
    tire_X_test = build_features(test_df, include_tire_age=True)

    tire_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
    )
    tire_model.fit(tire_X_train, y_train)
    tire_pred = tire_model.predict(tire_X_test)

    tire_metrics = evaluate_model(y_test, tire_pred)

    print("\nModel comparison")
    metrics = pd.DataFrame(
        {
            "Model": ["Baseline", "Tire-age"],
            "RMSE": [baseline_metrics["RMSE"], tire_metrics["RMSE"]],
            "MAE": [baseline_metrics["MAE"], tire_metrics["MAE"]],
        }
    )

    print(metrics.to_string(index=False))

    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print(f"\nSaved metrics to: {OUTPUT_DIR / 'model_metrics.csv'}")

    plot_one_stint(test_df, baseline_model, tire_model)

    print("\nDone. Open outputs/predicted_vs_actual_stint.png to inspect the plot.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
