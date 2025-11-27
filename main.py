#!/usr/bin/env python3
"""
F1 PIT STOP STRATEGY OPTIMIZER - MULTI-RACE, EXTENDED FEATURES
Adds simple condition-related features (avg_pit_lap, laps_completed).
"""

import os
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, r2_score
import scipy.stats as stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("F1 PIT STOP STRATEGY OPTIMIZER (MULTI-RACE, EXTENDED)")
print("Vajra Roshni Akurathi | 2210110779")
print("=" * 70)

# 1. SETUP
cache_dir = "./f1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
sns.set_style("whitegrid")

# 2. BUILD LIST OF 2024 RACES
print("\nBuilding 2024 race list...")
schedule_2024 = fastf1.get_event_schedule(2024)
gp_names = schedule_2024["EventName"].tolist()
print(f"Found {len(gp_names)} events in 2024:")
print(gp_names)

# 3. LOAD ALL 2024 RACE SESSIONS
print("\nLoading 2024 race data (first run can take time, later uses cache)...")

all_laps = []
all_results = []
all_pits = []

for gp in tqdm(gp_names, desc="Races"):
    try:
        race = fastf1.get_session(2024, gp, "R")
        race.load()

        # for extra features later
        laps_gp = race.laps.copy()
        laps_gp["EventName"] = gp
        all_laps.append(laps_gp)

        res_gp = race.results[["Abbreviation", "Position"]].copy()
        res_gp["EventName"] = gp
        all_results.append(res_gp)

        pits_gp = race.laps[race.laps["PitOutTime"].notnull()].copy()
        pits_gp["EventName"] = gp
        all_pits.append(pits_gp)

        print(f"  OK: {gp}")
    except Exception as e:
        print(f"  SKIPPED {gp} (error: {e})")

laps = pd.concat(all_laps, ignore_index=True)
results = pd.concat(all_results, ignore_index=True)
pit_stops = pd.concat(all_pits, ignore_index=True)

print(f"\nLoaded {len(laps)} laps, {len(pit_stops)} pit stops, {len(results)} driver results")

# 4. FEATURE ENGINEERING
print("\nEngineering features ...")

# basic: number of stops per driver
pit_counts = (
    pit_stops.groupby("Driver")["LapNumber"]
    .count()
    .reset_index()
    .rename(columns={"LapNumber": "num_stops"})
)

# average pit lap per driver (stint timing proxy)
pit_avg_lap = (
    pit_stops.groupby("Driver")["LapNumber"]
    .mean()
    .reset_index()
    .rename(columns={"LapNumber": "avg_pit_lap"})
)

# laps completed per driver (how long they stayed in the race)
laps_completed = (
    laps.groupby("Driver")["LapNumber"]
    .max()
    .reset_index()
    .rename(columns={"LapNumber": "laps_completed"})
)

# merge all features with results
df = results.merge(pit_counts, left_on="Abbreviation", right_on="Driver", how="left")
df = df.merge(pit_avg_lap, on="Driver", how="left")
df = df.merge(laps_completed, on="Driver", how="left")

df[["num_stops", "avg_pit_lap", "laps_completed"]] = df[
    ["num_stops", "avg_pit_lap", "laps_completed"]
].fillna(0)

df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
df["top5"] = (df["Position"] <= 5).astype(int)

print("\nSample of engineered dataset:")
print(df[["Abbreviation", "Position", "num_stops", "avg_pit_lap",
          "laps_completed", "top5"]].head(10))

# 5. EDA
print("\nCreating EDA plot...")

plt.figure(figsize=(12, 8))

# Scatter: num_stops vs position
plt.subplot(2, 2, 1)
sc = plt.scatter(
    df["num_stops"],
    df["Position"],
    c=df["top5"],
    cmap="RdYlGn_r",
    s=80,
    alpha=0.8
)
plt.xlabel("Number of Pit Stops")
plt.ylabel("Finishing Position")
plt.title("Pit Stops vs Finish Position (Green = Top 5)")
plt.gca().invert_yaxis()
plt.colorbar(sc, label="Top 5 = 1, Others = 0")

# Histogram of avg pit lap
plt.subplot(2, 2, 2)
plt.hist(df["avg_pit_lap"], bins=20, edgecolor="black")
plt.xlabel("Average Pit Lap")
plt.ylabel("Frequency")
plt.title("Distribution of Average Pit Lap")

# Boxplot: Top5 vs Others (num_stops)
plt.subplot(2, 2, 3)
top5_stops = df[df["top5"] == 1]["num_stops"]
other_stops = df[df["top5"] == 0]["num_stops"]
plt.boxplot([top5_stops, other_stops], labels=["Top 5", "Others"])
plt.ylabel("Number of Pit Stops")
plt.title("Pit Strategy: Top 5 vs Others (num_stops)")

# Scatter: avg_pit_lap vs Position
plt.subplot(2, 2, 4)
plt.scatter(df["avg_pit_lap"], df["Position"], alpha=0.7)
plt.xlabel("Average Pit Lap")
plt.ylabel("Finishing Position")
plt.title("Average Pit Lap vs Finish Position")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("eda_extended.png", dpi=300, bbox_inches="tight")
print("Saved EDA figure as eda_extended.png")
plt.close()


# 6. STATISTICAL TESTS (num_stops for clarity)
print("\nStatistical analysis (using num_stops groups):")

groups = [g["Position"].dropna() for _, g in df.groupby("num_stops")]
valid_groups = [g for g in groups if len(g) > 1]

if len(valid_groups) > 1:
    f_stat, p_anova = stats.f_oneway(*valid_groups)
    print(f"  ANOVA (pit stops -> position): F = {f_stat:.3f}, p = {p_anova:.3e}")
else:
    p_anova = None
    print("  ANOVA: not enough data for multiple groups.")

if len(top5_stops) > 1 and len(other_stops) > 1:
    t_stat, p_ttest = stats.ttest_ind(top5_stops, other_stops, equal_var=False)
    print(f"  T-test (Top5 vs Others, num_stops): t = {t_stat:.3f}, p = {p_ttest:.3e}")
else:
    p_ttest = None
    print("  T-test: not enough data for two groups.")

# 7. MACHINE LEARNING – COMPARE SIMPLE VS EXTENDED FEATURES
print("\nTraining machine learning models...")

# SIMPLE: only num_stops
X_simple = df[["num_stops"]].values

# EXTENDED: num_stops + avg_pit_lap + laps_completed
X_extended = df[["num_stops", "avg_pit_lap", "laps_completed"]].values

y_reg = df["Position"].fillna(df["Position"].median()).values
y_clf = df["top5"].values

def train_and_report(X, label):
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    reg = LinearRegression().fit(X_train, y_reg_train)
    r2 = r2_score(y_reg_test, reg.predict(X_test))

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_clf_train)
    y_pred = rf.predict(X_test)
    report = classification_report(y_clf_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n=== {label} FEATURES ===")
    print(f"  Linear Regression R^2: {r2:.3f}")
    print(f"  RF Top-5 precision: {report['1']['precision']:.3f}")
    print(f"  RF Top-5 recall:    {report['1']['recall']:.3f}")
    print(f"  RF Top-5 F1-score:  {report['1']['f1-score']:.3f}")
    return r2, report["1"]["f1-score"]

r2_simple, f1_simple = train_and_report(X_simple, "SIMPLE (num_stops only)")
r2_ext, f1_ext = train_and_report(X_extended, "EXTENDED (num_stops + avg_pit_lap + laps_completed)")

# 8. SIMPLE STRATEGY OPTIMISATION
print("\nRunning simple pit stop optimization toy model...")

def simulate_strategy(num_stops_optimal, baseline_time=5400.0, stop_time_loss=25.0):
    current_stops = 2
    delta_stops = current_stops - num_stops_optimal
    optimized_time = baseline_time - max(delta_stops, 0) * stop_time_loss * 0.9
    gain = baseline_time - optimized_time
    return optimized_time, gain

optimal_stops = 1
opt_time, gain = simulate_strategy(optimal_stops)

print(f"  Baseline strategy: 2 stops -> 5400.0 s (assumed)")
print(f"  Hypothetical optimal: {optimal_stops} stop -> {opt_time:.1f} s")
print(f"  Estimated time gain: {gain:.1f} s")

# 9. SUMMARY
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Drivers analyzed: {len(df)}")
print(f"Races used: {len(gp_names)} from 2024 season")
print(f"ANOVA p-value (stops -> position): {p_anova:.3e}" if p_anova is not None else "ANOVA: N/A")
print(f"T-test p-value (Top5 vs Others stops): {p_ttest:.3e}" if p_ttest is not None else "T-test: N/A")
print(f"Simple R^2 (num_stops): {r2_simple:.3f}, Extended R^2: {r2_ext:.3f}")
print(f"Simple Top-5 F1: {f1_simple:.3f}, Extended Top-5 F1: {f1_ext:.3f}")
print("EDA figure: eda_extended.png")
print(f"Toy optimization: gain ≈ {gain:.1f} seconds by reducing to {optimal_stops} stop(s)")
print("=" * 70)
print("Done.")
