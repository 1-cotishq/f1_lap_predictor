# Formula One Lap Time Predictor

This project is a beginner-friendly example of how to build a lap-time prediction model for one Formula 1 race using the Kaggle dataset `Formula 1 World Championship (1950 to 2020)`.

## What this project does

- Chooses one dry race with no red flags and 5 to 10 finished drivers
- Cleans the lap data
- Splits the data by stint, not randomly
- Compares two models:
  - Baseline model
  - Tire-age model
- Reports RMSE and MAE
- Saves a plot for one full stint showing predicted vs actual lap times

## Windows setup in VS Code

1. Open the workspace folder in VS Code.
2. Open the integrated terminal with **Terminal > New Terminal**.
3. Run these commands:

```powershell
cd "c:\Users\gupta\OneDrive\Desktop\f1_lap_predictor"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `.venv` does not exist, create it first:

```powershell
py -m venv .venv
```

If VS Code does not use the virtual environment automatically, select it from the Python interpreter menu:

- Press Ctrl+Shift+P
- Choose `Python: Select Interpreter`
- Pick the interpreter inside `.venv\Scripts\python.exe`

## Download the Kaggle CSV files

Download the dataset `Formula 1 World Championship (1950 to 2020)` from Kaggle and place these files into a folder named `data` inside this workspace:

- `lap_times.csv`
- `pit_stops.csv`
- `results.csv`
- `races.csv`

Example path:

```text
c:\Users\gupta\OneDrive\Desktop\f1_lap_predictor\data\lap_times.csv
```

Do not rename the files. The script expects the column names from the original
Kaggle dataset.

## Run the project

```powershell
cd "c:\Users\gupta\OneDrive\Desktop\f1_lap_predictor"
.\.venv\Scripts\Activate.ps1
python src/f1_lap_predictor.py
```

The script prints the number of removed laps during cleaning, the selected
race, the selected finishers, the stint-based train/test row counts, and the
RMSE/MAE comparison.

### Choosing the race

The script chooses the latest race with at least five classified finishers so
it can run immediately on the static CSV files. Because the standard Kaggle
`races.csv` usually does not record weather or red flags, verify that the
printed race was dry and had no red flag. For a fixed race in your report,
open `src/f1_lap_predictor.py`, set `RACE_ID` to that race's `raceId`, and run
the script again.

## Output files

After running the script, you will get:

- `outputs/model_metrics.csv`
- `outputs/predicted_vs_actual_stint.png`
- `outputs/cleaning_summary.csv`

## Beginner notes

- `lap_times.csv` contains lap-by-lap timing information.
- `pit_stops.csv` tells us when each driver stopped for tires.
- `results.csv` helps identify which drivers finished the race.
- `races.csv` helps find the race we want to study.
- A stint starts at the first lap of the race and after each pit stop. The
  first model uses race lap, one-hot encoded driver, stint number, and laps
  remaining as a simple fuel-load/race-progress proxy; the second adds tire
  age and tire age squared.
- Driver IDs are treated as categories rather than numeric measurements. The
  same one-hot encoded feature columns are aligned between training, testing,
  and plotting.
- Complete stints remain separated chronologically for evaluation. This
  preserves the assignment's stint-based split while the `laps_remaining`
  feature helps the models account for fuel burn and race progress.

## Important reminder

The project uses one race only. That keeps the track and weather conditions stable, which makes the model easier to understand.
