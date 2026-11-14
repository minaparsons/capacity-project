import numpy as np
import pandas as pd
import pytest

from src.demand import LOG_COLUMNS, choose_model, forecast_demand, run_monthly_forecast

MAE = {"naive": 10.0, "seasonal_naive": 8.0,"damped_ets": 9.5}


def history(n=22, start="2025-01-01"):
    rng = np.random.default_rng(2)
    t = np.arange(n)
    values = 100 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, n)
    return pd.Series(values, index=pd.date_range(start, periods=n, freq="MS"))


# ---------- choose_model ----------

def test_no_previous_picks_lowest_mae():
    assert choose_model(MAE) == "seasonal_naive"


def test_keeps_previous_when_within_margin():
    # 8.0 is only ~16% better than 9.5, so a 20% margin keeps damped_ets
    assert choose_model(MAE, "damped_ets", switch_margin=0.2) == "damped_ets"


def test_switches_when_clearly_beaten():
    # 8.0 is ~16% better than 9.5, beyond a 10% margin
    assert choose_model(MAE, "damped_ets", switch_margin=0.1) == "seasonal_naive"


def test_unknown_previous_is_ignored():
    assert choose_model(MAE, "retired_model") == "seasonal_naive"


def test_tie_at_zero_keeps_previous():
    assert choose_model({"a": 0.0, "b": 0.0}, "b") == "b"


# ---------- forecast_demand with a previous model ----------

def test_forecast_reports_used_and_best_model_separately():
    result = forecast_demand(history(), h=1, previous_model="naive", switch_margin=1.0)
    assert result.model == "naive"  # margin of 100% never switches
    assert result.best_by_mae == min(result.backtest_mae, key=result.backtest_mae.get)


# ---------- run_monthly_forecast and the log ----------

def test_first_run_creates_log_with_header(tmp_path):
    log_path = tmp_path / "logs" / "model_log.csv"
    result = run_monthly_forecast(history(), h=1, series_name="maths_0", log_path=log_path)

    log = pd.read_csv(log_path)
    assert list(log.columns) == LOG_COLUMNS
    assert len(log) == 1
    assert log.loc[0, "last_month"] == "2026-10"
    assert log.loc[0, "model"] == result.best_by_mae == result.model


def test_log_records_forecast_and_settings(tmp_path):
    log_path = tmp_path / "model_log.csv"
    result = run_monthly_forecast(history(), h=1, series_name="maths_0", log_path=log_path,
                                  coverage=0.9, switch_margin=0.2)

    row = pd.read_csv(log_path).iloc[0]
    for column in ("forecast", "lower", "upper"):
        assert row[column] == pytest.approx(result.forecast[column].iloc[0])
    assert row["coverage"] == 0.9
    assert row["switch_margin"] == 0.2
    assert row["n_windows"] == 7  # 22 months, growth model eligible: origins 15..21


def test_next_month_uses_previous_model(tmp_path):
    log_path = tmp_path / "model_log.csv"
    run_monthly_forecast(history(21), h=1, series_name="maths_0", log_path=log_path,
                         switch_margin=1.0)
    first = pd.read_csv(log_path).loc[0, "model"]

    result = run_monthly_forecast(history(22), h=1, series_name="maths_0",
                                  log_path=log_path, switch_margin=1.0)

    assert result.model == first
    assert len(pd.read_csv(log_path)) == 2


def test_rerun_same_month_ignores_its_own_row(tmp_path):
    log_path = tmp_path / "model_log.csv"
    row = dict.fromkeys(LOG_COLUMNS, 1)
    row.update(series="maths_0", last_month="2026-10", model="naive")
    pd.DataFrame([row]).to_csv(log_path, index=False)

    result = run_monthly_forecast(history(), h=1, series_name="maths_0",
                                  log_path=log_path, switch_margin=1.0)

    assert result.model == result.best_by_mae


@pytest.mark.parametrize("other_series", ["english_0", "maths_1"])
def test_previous_model_is_per_series(tmp_path, other_series):
    log_path = tmp_path / "model_log.csv"
    row = dict.fromkeys(LOG_COLUMNS, 1)
    row.update(series=other_series, last_month="2026-09", model="naive")
    pd.DataFrame([row]).to_csv(log_path, index=False)

    result = run_monthly_forecast(history(), h=1, series_name="maths_0",
                                  log_path=log_path, switch_margin=1.0)

    assert result.model == result.best_by_mae
