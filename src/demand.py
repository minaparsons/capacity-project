
#for working out predicted demand
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON_LENGTH = 12
MIN_TRAIN = 12  # one full season: the least seasonal_naive needs
MIN_WINDOWS = 6  # backtest windows a model needing extra history must get before it is compared
MIN_GROWTH_PAIRS = 3  # matched year-on-year months needed to estimate growth


@dataclass
class Demand:
    demand: dict = field(default_factory=dict)

    def get_subject_demand(self, subject, gender):
        return self.demand[subject.lower()][gender]


@dataclass
class ForecastResult:
    model: str  # model actually used (may be last month's, see choose_model)
    best_by_mae: str  # lowest-MAE model this run
    backtest_mae: dict
    forecast: pd.DataFrame  # columns: forecast, lower, upper; monthly index


# ---------- loading and validation ----------

def load_series(path, date_col="month", value_col="hours"):
    """Read a monthly demand CSV into a validated Series indexed by month start."""
    df = pd.read_csv(path)
    for col in (date_col, value_col):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
    dates = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    series = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values, index=dates)
    return validate_series(series.sort_index())


def validate_series(series, min_length=MIN_TRAIN + 1):
    """Check the series is complete, monthly, non-negative and long enough."""
    if series.isna().any():
        raise ValueError("series contains missing or non-numeric values")
    if (series < 0).any():
        raise ValueError("series contains negative values")
    if series.index.duplicated().any():
        raise ValueError("series contains duplicate months")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    if len(series) != len(expected) or not (series.index == expected).all():
        raise ValueError("series has missing months or dates not at month start")
    if len(series) < min_length:
        raise ValueError(f"need at least {min_length} months, got {len(series)}")
    return series.asfreq("MS").astype(float)


# ---------- models: each takes a training array and horizon, returns h forecasts ----------

def naive(y, h):
    return np.repeat(y[-1], h)


def seasonal_naive(y, h):
    last_season = y[-SEASON_LENGTH:]
    return np.resize(last_season, h)


def yoy_growth(y):
    """Total of the recent months / total of the same months a year earlier.

    Uses up to the last 12 months that have a value a year before. Returns 1.0
    (no growth) if there are fewer than MIN_GROWTH_PAIRS such months or the base is zero.
    """
    k = min(SEASON_LENGTH, len(y) - SEASON_LENGTH)
    if k < MIN_GROWTH_PAIRS:
        return 1.0
    recent = y[-k:].sum()
    year_before = y[-k - SEASON_LENGTH:-SEASON_LENGTH].sum()
    return recent / year_before if year_before > 0 else 1.0


def seasonal_naive_growth(y, h):
    """Same month last year, scaled by year-on-year growth (compounded beyond 12 months)."""
    years_ahead = np.arange(h) // SEASON_LENGTH + 1
    return seasonal_naive(y, h) * yoy_growth(y) ** years_ahead


def linear_trend(y, h):
    t = np.arange(len(y))
    slope, intercept = np.polyfit(t, y, 1)
    return intercept + slope * np.arange(len(y), len(y) + h)


def damped_ets(y, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
    return np.asarray(fit.forecast(h))


def holt_winters(y, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=SEASON_LENGTH, initialization_method="heuristic",
        ).fit()
    return np.asarray(fit.forecast(h))


MODELS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_growth": seasonal_naive_growth,
    "linear_trend": linear_trend,
    "damped_ets": damped_ets,
    "holt_winters": holt_winters,
}

# Models needing more training history than MIN_TRAIN.
# seasonal_naive_growth: below this it falls back to plain seasonal naive, so
#   backtesting it there would judge it on windows that don't reflect its real use.
# holt_winters: needs two full seasons to initialise its seasonal pattern.
MODEL_MIN_TRAIN = {
    "seasonal_naive_growth": SEASON_LENGTH + MIN_GROWTH_PAIRS,
    "holt_winters": 2 * SEASON_LENGTH,
}


# ---------- backtesting, selection and forecasting ----------

def backtest(y, model, h, min_train=MIN_TRAIN):
    """Rolling-origin errors (actual - forecast), shape (n_origins, h)."""
    y = np.asarray(y, dtype=float)
    origins = range(min_train, len(y) - h + 1)
    if len(origins) == 0:
        raise ValueError(f"need at least {min_train + h} months to backtest h={h}")
    return np.array([y[o:o + h] - model(y[:o], h) for o in origins])


def eligible_models(n, h, min_train=MIN_TRAIN):
    """Models with enough history to compare, and the shared first backtest origin.

    A model needing extra history joins only once it gets MIN_WINDOWS backtest windows;
    all eligible models are then backtested from the same origins so MAEs are comparable.
    """
    needed = {name: max(min_train, MODEL_MIN_TRAIN.get(name, 0)) for name in MODELS}
    names = [name for name in MODELS
             if name not in MODEL_MIN_TRAIN or n - h + 1 - needed[name] >= MIN_WINDOWS]
    return names, max(needed[name] for name in names)


def select_model(y, h, min_train=MIN_TRAIN):
    """Backtest every eligible model; return (best name, {name: MAE}, {name: errors})."""
    names, start = eligible_models(len(y), h, min_train)
    errors = {name: backtest(y, MODELS[name], h, start) for name in names}
    mae = {name: float(np.mean(np.abs(e))) for name, e in errors.items()}
    best = min(mae, key=mae.get)
    return best, mae, errors


def choose_model(mae, previous=None, switch_margin=0.1):
    """Lowest-MAE model, unless `previous` is within switch_margin of it (then keep previous)."""
    best = min(mae, key=mae.get)
    if previous in mae and mae[best] >= (1 - switch_margin) * mae[previous]:
        return previous
    return best


def forecast_demand(series, h, coverage=0.8, min_train=MIN_TRAIN,
                    previous_model=None, switch_margin=0.1):
    """Choose a model (see choose_model) and forecast h months with empirical intervals."""
    series = validate_series(series, min_length=min_train + h)
    y = series.to_numpy()
    best, mae, errors = select_model(y, h, min_train)
    model = choose_model(mae, previous_model, switch_margin)

    point = MODELS[model](y, h)
    alpha = (1 - coverage) / 2
    lower = point + np.quantile(errors[model], alpha, axis=0)
    upper = point + np.quantile(errors[model], 1 - alpha, axis=0)

    index = pd.date_range(series.index[-1], periods=h + 1, freq="MS")[1:]
    table = pd.DataFrame(
        {"forecast": point, "lower": lower, "upper": upper}, index=index
    ).clip(lower=0)
    return ForecastResult(model=model, best_by_mae=best, backtest_mae=mae, forecast=table)


# ---------- monthly run with model log ----------

LOG_COLUMNS = (["run_date", "series", "last_month", "horizon", "model", "best_by_mae"]
               + [f"mae_{name}" for name in MODELS])


def read_previous_model(log_path, series_name, last_month):
    """Model used for this series in the most recent run on data before last_month."""
    path = Path(log_path)
    if not path.exists():
        return None
    log = pd.read_csv(path, dtype=str)
    earlier = log[(log["series"] == series_name) & (log["last_month"] < last_month)]
    return None if earlier.empty else earlier["model"].iloc[-1]


def append_log(log_path, series_name, last_month, h, result):
    row = {"run_date": date.today().isoformat(), "series": series_name,
           "last_month": last_month, "horizon": h, "model": result.model,
           "best_by_mae": result.best_by_mae}
    row.update({f"mae_{name}": round(v, 4) for name, v in result.backtest_mae.items()})
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False)


def run_monthly_forecast(series, h, series_name, log_path, coverage=0.8, switch_margin=0.1):
    """Forecast one series, sticking with last month's model unless clearly beaten, and log it."""
    last_month = series.index.max().strftime("%Y-%m")
    previous = read_previous_model(log_path, series_name, last_month)
    result = forecast_demand(series, h, coverage,
                             previous_model=previous, switch_margin=switch_margin)
    append_log(log_path, series_name, last_month, h, result)
    return result
