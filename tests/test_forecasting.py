import numpy as np
import pandas as pd
import pytest

from src.demand import (
    MIN_WINDOWS,
    backtest,
    damped_ets,
    eligible_models,
    forecast_demand,
    holt_winters,
    linear_trend,
    load_series,
    naive,
    seasonal_naive,
    seasonal_naive_growth,
    select_model,
    validate_series,
    yoy_growth,
)


def monthly(values, start="2022-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def seasonal_values(n=48):
    t = np.arange(n)
    return 100 + 20 * np.sin(2 * np.pi * t / 12)


def trend_values(n=48):
    return 50 + 2.0 * np.arange(n)


# ---------- loading and validation ----------

def test_load_series_reads_csv_and_normalises_dates(tmp_path):
    path = tmp_path / "demand.csv"
    dates = pd.date_range("2022-01-15", periods=30, freq="MS") + pd.Timedelta(days=14)
    pd.DataFrame({"month": dates.strftime("%Y-%m-%d"), "hours": range(30)}).to_csv(path, index=False)

    series = load_series(path)

    assert len(series) == 30
    assert series.index[0] == pd.Timestamp("2022-02-01")
    assert series.index.freqstr == "MS"


def test_load_series_rejects_missing_column(tmp_path):
    path = tmp_path / "demand.csv"
    pd.DataFrame({"month": ["2022-01-01"], "demand": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_series(path)


def test_validate_accepts_clean_series():
    assert len(validate_series(monthly(trend_values(30)))) == 30


@pytest.mark.parametrize(
    "series, message",
    [
        (monthly([1.0] * 29 + [np.nan]), "missing or non-numeric"),
        (monthly([1.0] * 29 + [-1.0]), "negative"),
        (monthly([1.0] * 10), "at least"),
    ],
)
def test_validate_rejects_bad_values(series, message):
    with pytest.raises(ValueError, match=message):
        validate_series(series)


def test_validate_rejects_gap():
    series = monthly([1.0] * 30).drop(pd.Timestamp("2023-01-01"))
    with pytest.raises(ValueError, match="missing months"):
        validate_series(series)


def test_validate_rejects_duplicate_month():
    series = monthly([1.0] * 30)
    series = pd.concat([series, series.iloc[:1]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        validate_series(series)


# ---------- models ----------

def test_naive_repeats_last_value():
    assert naive(np.array([1.0, 2.0, 5.0]), 3).tolist() == [5.0, 5.0, 5.0]


def test_seasonal_naive_repeats_last_year():
    y = np.arange(24, dtype=float)
    assert seasonal_naive(y, 14).tolist() == list(range(12, 24)) + [12.0, 13.0]


def growing_seasonal_values(n=22, growth=1.2):
    t = np.arange(n)
    return (100 + 20 * np.sin(2 * np.pi * t / 12)) * growth ** (t / 12)


def test_yoy_growth_matches_true_growth():
    assert yoy_growth(growing_seasonal_values(22)) == pytest.approx(1.2)
    assert yoy_growth(growing_seasonal_values(40)) == pytest.approx(1.2)


@pytest.mark.parametrize("y", [np.arange(1.0, 15.0), np.r_[np.zeros(12), np.ones(10)]])
def test_yoy_growth_falls_back_to_one(y):
    # first: only 2 matched months; second: zero base a year earlier
    assert yoy_growth(y) == 1.0


def test_seasonal_naive_growth_scales_last_year():
    first_year = np.arange(1.0, 13.0)
    y = np.r_[first_year, 1.2 * first_year]
    forecast = seasonal_naive_growth(y, 13)
    np.testing.assert_allclose(forecast[:3], 1.44 * first_year[:3])
    np.testing.assert_allclose(forecast[12], 1.2 ** 3 * first_year[0])  # compounded


def test_linear_trend_extrapolates_line():
    np.testing.assert_allclose(linear_trend(trend_values(30), 3), [110.0, 112.0, 114.0])


def test_damped_ets_flat_series_stays_flat():
    np.testing.assert_allclose(damped_ets(np.full(30, 40.0), 3), 40.0, atol=1e-6)


def test_holt_winters_follows_trend_plus_season():
    t = np.arange(39)
    y = 100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12)
    np.testing.assert_allclose(holt_winters(y[:36], 3), y[36:], rtol=0.02)


def test_holt_winters_needs_two_seasons():
    with pytest.raises(ValueError):
        holt_winters(seasonal_values(23), 1)


# ---------- eligibility ----------

def test_with_22_months_growth_model_joins_but_holt_winters_does_not():
    names, start = eligible_models(22, h=1)  # 22 - 15 = 7 windows for the growth model
    assert "seasonal_naive_growth" in names
    assert "holt_winters" not in names
    assert start == 15


def test_with_22_months_and_h3_growth_model_waits():
    names, start = eligible_models(22, h=3)  # 22 - 3 + 1 - 15 = 5 windows, below MIN_WINDOWS
    assert "seasonal_naive_growth" not in names
    assert start == 12


def test_holt_winters_joins_once_it_has_enough_windows():
    joins_at = 24 + MIN_WINDOWS  # h=1: windows = n - 24
    assert "holt_winters" not in eligible_models(joins_at - 1, h=1)[0]
    names, start = eligible_models(joins_at, h=1)
    assert "holt_winters" in names
    assert start == 24  # every model now backtested from the same origin


def test_all_models_share_backtest_origins():
    _, _, errors = select_model(seasonal_values(36), h=1)
    assert len({e.shape for e in errors.values()}) == 1


# ---------- backtesting and selection ----------

def test_backtest_shape_and_zero_error_for_exact_model():
    errors = backtest(trend_values(30), linear_trend, h=3, min_train=24)
    assert errors.shape == (4, 3)  # origins 24..27
    np.testing.assert_allclose(errors, 0, atol=1e-9)


def test_backtest_rejects_too_short_series():
    with pytest.raises(ValueError, match="backtest"):
        backtest(trend_values(25), naive, h=3, min_train=24)


def test_select_model_picks_seasonal_model_for_seasonal_data():
    # no growth, so both seasonal naive variants forecast (almost) identically
    best, mae, _ = select_model(seasonal_values(), h=3)
    assert best in {"seasonal_naive", "seasonal_naive_growth"}
    assert set(mae) == {"naive", "seasonal_naive", "seasonal_naive_growth",
                        "linear_trend", "damped_ets", "holt_winters"}


def test_select_model_picks_growth_model_for_growing_seasonal_data_now():
    best, mae, _ = select_model(growing_seasonal_values(22), h=1)
    assert best == "seasonal_naive_growth"
    assert "holt_winters" not in mae


def test_select_model_picks_holt_winters_for_growing_seasonal_data():
    t = np.arange(48)
    best, _, _ = select_model(100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12), h=1)
    assert best == "holt_winters"


def test_select_model_picks_linear_trend_for_straight_line():
    best, _, _ = select_model(trend_values(), h=3)
    assert best == "linear_trend"


# ---------- end-to-end forecast ----------

def test_forecast_continues_monthly_index_with_ordered_intervals():
    rng = np.random.default_rng(0)
    series = monthly(seasonal_values() + rng.normal(0, 3, 48))

    result = forecast_demand(series, h=6)
    table = result.forecast

    assert list(table.index) == list(pd.date_range("2026-01-01", periods=6, freq="MS"))
    assert (table["lower"] <= table["forecast"]).all()
    assert (table["forecast"] <= table["upper"]).all()
    assert result.backtest_mae[result.model] == min(result.backtest_mae.values())


def test_forecast_intervals_collapse_for_perfect_fit():
    table = forecast_demand(monthly(trend_values()), h=3).forecast
    np.testing.assert_allclose(table["upper"] - table["lower"], 0, atol=1e-9)


@pytest.mark.parametrize("h", [1, 3])
def test_forecast_works_with_22_months_of_history(h):
    # Jan 2025 - Oct 2026: the real data available
    rng = np.random.default_rng(1)
    series = monthly(seasonal_values(22) + rng.normal(0, 3, 22), start="2025-01-01")

    table = forecast_demand(series, h=h).forecast

    assert table.index[0] == pd.Timestamp("2026-11-01")
    assert len(table) == h
    assert (table["lower"] <= table["upper"]).all()


def test_forecast_never_negative():
    falling = monthly(np.maximum(200 - 8.0 * np.arange(30), 0))
    table = forecast_demand(falling, h=6).forecast
    assert (table >= 0).all().all()
