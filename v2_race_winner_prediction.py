#!/usr/bin/env python3
import fastf1, pandas as pd, numpy as np, datetime as dt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

# ── cache ─────────────────────────────────────────────────────
Path("f1_cache").mkdir(exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# ── 1. helper: next race on the calendar ──────────────────────
def next_race_event(today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    for yr in (today.year, today.year + 1): # handle winter break
        sched = fastf1.get_event_schedule(yr, include_testing=False)
        future = sched[sched["EventDate"].dt.date > today]
        if not future.empty:
            row = future.iloc[0]
            return dict(
                year=yr,
                round=int(row["RoundNumber"]),
                name=row["EventName"],
                date=row["EventDate"].date(),
            )
    raise RuntimeError("No upcoming F1 race found")

# helper – last completed race --------------------------------
def last_completed_race():
    today = dt.date.today()
    for yr in (today.year, today.year - 1):
        sched = fastf1.get_event_schedule(yr, include_testing=False)
        past  = sched[sched["EventDate"].dt.date < today]
        if not past.empty:
            row = past.iloc[-1]
            return dict(year=yr, round=int(row["RoundNumber"]),
                        name=row["EventName"], date=row["EventDate"].date())
    raise RuntimeError("No past race found")

nr = last_completed_race()
print(f"→ Back-test up to {nr['name']} ({nr['date']})")
# nr = next_race_event()
# print(f"→ Next race: {nr['name']}  on  {nr['date']}  (Round {nr['round']})")

# ── collect all completed races up to that date ---------------
rows = []
for yr in (2024, 2025):
    sched = fastf1.get_event_schedule(yr, include_testing=False)
    for rnd, ev_date in sched[["RoundNumber", "EventDate"]].itertuples(index=False):
        if ev_date.date() >= nr["date"]:
            break
        s = fastf1.get_session(yr, rnd, "R"); s.load()
        r = s.results.assign(
                season  = yr,
                round   = rnd,
                circuit = s.event["Location"],
                date    = ev_date.date(),        # ← add this field
        )
        for seg in ("Q1", "Q2", "Q3"):
            r[seg] = pd.to_timedelta(r[seg]).dt.total_seconds()
        r["bestQual"] = r[["Q1", "Q2", "Q3"]].min(axis=1)
        r["win"] = (r["Position"] == 1).astype(int)
        rows.append(r)

df = pd.concat(rows, ignore_index=True)

# ── features + model set-up -----------------------------------
feat_num = ["GridPosition", "bestQual"]
# bestQual (raw lap time) shows pure speed on Saturday.
# GridPosition is the actual starting slot after penalties, parc-fermé violations, sprint-qualifying re-ordering, etc.

feat_cat = ["DriverId", "TeamId", "circuit"]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), feat_num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore")),
    ]), feat_cat)
])
model = Pipeline([("pre", pre),
                  ("gb", GradientBoostingClassifier(random_state=0))])

# ── walk-forward back-test ------------------------------------
model_hits, pole_hits, loglosses = [], [], []

for (yr, rnd), race_df in (
        df.sort_values(["season", "round"]).groupby(["season", "round"])):

    train_df = df[(df["season"] < yr) | ((df["season"] == yr) & (df["round"] < rnd))]
    if train_df.empty:                  # first race has no history
        continue

    model.fit(train_df[feat_num + feat_cat], train_df["win"])

    X_race = race_df[feat_num + feat_cat]
    y_race = race_df["win"].values
    proba  = model.predict_proba(X_race)[:, 1]

    pred_winner = race_df.iloc[proba.argmax()]["Abbreviation"]
    real_winner = race_df.loc[race_df["win"] == 1, "Abbreviation"].iloc[0]

    pole_row = race_df[race_df["GridPosition"] == 1]
    if pole_row.empty:                                   # edge-case fallback
        pole_row = race_df.loc[[race_df["bestQual"].idxmin()]]
    pole_winner = pole_row.iloc[0]["Abbreviation"]

    model_hits.append(pred_winner == real_winner)
    pole_hits.append(pole_winner  == real_winner)
    loglosses.append(log_loss(y_race, proba))

# print(f"\nModel hit-rate : {np.mean(model_hits):.3f}")
# print(f"P1 baseline    : {np.mean(pole_hits):.3f}")
# print(f"Mean log-loss  : {np.mean(loglosses):.4f}")

# ── winner prediction for *that* last race --------------------
train_df = df[df["date"] < nr["date"]]        # train on races before Austria
model.fit(train_df[feat_num + feat_cat], train_df["win"])

# use qualifying (Q or SQ) for the last GP
qual = None
for kind in ("Q", "SQ"):
    try:
        qs = fastf1.get_session(nr["year"], nr["round"], kind); qs.load()
        if not qs.results.empty:
            qual = qs
            break
    except Exception:
        pass
if qual is None:
    raise RuntimeError("Qualifying data missing for GP (unexpected).")

future = qual.results.copy()
for seg in ("Q1", "Q2", "Q3"):
    future[seg] = pd.to_timedelta(future[seg]).dt.total_seconds()
future["bestQual"]     = future[["Q1", "Q2", "Q3"]].min(axis=1)
future["GridPosition"] = future["Position"]
future["circuit"]      = qual.event["Location"]
future["DriverId"]     = future["DriverId"].astype(str)
future["TeamId"]       = future["TeamId"].astype(str)

future["P_win"] = model.predict_proba(future[feat_num + feat_cat])[:, 1]

print(f"\nModel hit-rate : {np.mean(model_hits):.3f}")
print(f"P1 baseline    : {np.mean(pole_hits):.3f}")
print(f"Mean log-loss  : {np.mean(loglosses):.4f}")

print(f"\n🏁  Predicted win probabilities – {nr['name']}")
print(
    future[["Abbreviation", "GridPosition", "bestQual", "P_win"]]
    .sort_values("P_win", ascending=False)
    .head(3)
    .to_string(index=False, formatters={"P_win": "{:.3f}".format})
)

