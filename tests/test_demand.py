# ---------- forecast_all_demand ----------

import numpy as np
import pandas as pd
import pytest

from src.demand import forecast_all_demand


def _history(level, n=22):
    t = np.arange(n)
    values = level + 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(level).normal(0, 2, n)
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=n, freq="MS"))


SERIES = {("Maths", 0): _history(100), ("maths", 1): _history(50), ("english", 0): _history(30)}


def test_builds_demand_for_every_subject_and_gender(tmp_path):
    demand, results = forecast_all_demand(SERIES, tmp_path / "log.csv")
    assert set(demand) == {"maths", "english"}
    assert set(demand["maths"]) == {0, 1}
    for (subject, gender), result in results.items():
        assert demand[subject][gender] == result.forecast["forecast"].iloc[0]


def test_upper_column_is_at_least_lower(tmp_path):
    lower, _ = forecast_all_demand(SERIES, tmp_path / "a.csv", column="lower")
    upper, _ = forecast_all_demand(SERIES, tmp_path / "b.csv", column="upper")
    for subject, by_gender in lower.items():
        for gender, hours in by_gender.items():
            assert upper[subject][gender] >= hours


def test_logs_one_row_per_series(tmp_path):
    log_path = tmp_path / "log.csv"
    forecast_all_demand(SERIES, log_path)
    assert set(pd.read_csv(log_path)["series"]) == {"maths_0", "maths_1", "english_0"}


def test_rejects_unknown_column(tmp_path):
    with pytest.raises(ValueError):
        forecast_all_demand(SERIES, tmp_path / "log.csv", column="median")


# ---------- load_subject_demand ----------

from src.demand import FEMALE, MALE, load_subject_demand


def _write_subject_csv(path, n=14, columns=("month", "female_hours", "male_hours")):
    months = pd.date_range("2025-01-01", periods=n, freq="MS").strftime("%Y-%m")
    data = {columns[0]: months, columns[1]: range(100, 100 + n), columns[2]: range(10, 10 + n)}
    data.update({c: 0 for c in columns[3:]})
    pd.DataFrame(data).to_csv(path, index=False)


def test_subject_file_splits_into_female_and_male_series(tmp_path):
    path = tmp_path / "Maths.csv"
    _write_subject_csv(path)
    series = load_subject_demand(path)
    assert set(series) == {("maths", FEMALE), ("maths", MALE)}
    assert series[("maths", FEMALE)].iloc[0] == 100  # second column is female
    assert series[("maths", MALE)].iloc[0] == 10  # third column is male


def test_subject_file_reads_columns_by_position_not_name(tmp_path):
    path = tmp_path / "english.csv"
    _write_subject_csv(path, columns=("Date", "F", "M"))
    assert load_subject_demand(path)[("english", FEMALE)].iloc[-1] == 113


def test_subject_file_with_wrong_column_count_is_rejected(tmp_path):
    path = tmp_path / "physics.csv"
    _write_subject_csv(path, columns=("month", "female", "male", "total"))
    with pytest.raises(ValueError, match="expected 3 columns"):
        load_subject_demand(path)


def test_subject_file_normalises_dates_to_month_start(tmp_path):
    path = tmp_path / "maths.csv"
    dates = pd.date_range("2022-01-01", periods=14, freq="MS")
    pd.DataFrame({"month": dates.strftime("%Y-%m-15"),"f": range(14), "m": range(14)}).to_csv(
        path, index=False)
    for series in load_subject_demand(path).values():
        assert series.index[0] == pd.Timestamp("2022-01-01")
        assert series.index.freqstr == "MS"


def test_subject_file_with_missing_month_is_rejected(tmp_path):
    path = tmp_path / "maths.csv"
    _write_subject_csv(path)
    df = pd.read_csv(path)
    df.drop(index=5).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing months"):
        load_subject_demand(path)


def test_subject_file_with_non_numeric_hours_is_rejected(tmp_path):
    path = tmp_path / "maths.csv"
    _write_subject_csv(path)
    df = pd.read_csv(path).astype({"male_hours": object})
    df.loc[3, "male_hours"] = "n/a"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_subject_demand(path)


def test_subject_file_ending_in_current_month_is_rejected(tmp_path):
    path = tmp_path / "maths.csv"
    this_month = pd.Timestamp.today().to_period("M").to_timestamp()
    months = pd.date_range(end=this_month, periods=14, freq="MS").strftime("%Y-%m")
    pd.DataFrame({"month": months, "f": range(14), "m": range(14)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not over yet"):
        load_subject_demand(path)


def test_loaded_files_feed_straight_into_forecasting(tmp_path):
    for name in ("Maths.csv", "English.csv"):
        _write_subject_csv(tmp_path / name, n=22)
    series = {**load_subject_demand(tmp_path / "Maths.csv"),
              **load_subject_demand(tmp_path / "English.csv")}
    demand, _ = forecast_all_demand(series, tmp_path / "log.csv")
    for subject in ("maths", "english"):
        for gender in (FEMALE, MALE):
            assert demand[subject][gender] >= 0
