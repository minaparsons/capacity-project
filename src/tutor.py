import math

# A tutor is just a row of (weekly hours available, subjects, gender) - no identity is kept.
# These helpers check a row and split its hours between subjects.


def clean_tutor_row(hours, subjects, gender):
    """Validate a (hours, subjects, gender) row; return it with subjects lower-cased and de-duplicated."""
    if gender not in (0, 1):
        raise ValueError(f"gender must be 0 or 1, got {gender!r}")
    if not isinstance(hours, int) or hours < 0:
        raise ValueError(f"hours must be a whole number >= 0, got {hours!r}")
    subjects = list(dict.fromkeys(s.strip().lower() for s in subjects))
    if not subjects:
        raise ValueError("tutor must teach at least one subject")
    return hours, subjects, gender


def split_hours(hours, weights):
    """Split whole hours between subjects in proportion to weights, rounding each share down.

    weights: {subject: weight >= 0}. If every weight is 0, hours are split equally.
    Rounding down means the parts can sum to less than hours.
    """
    total = sum(weights.values())
    if total == 0:
        weights = dict.fromkeys(weights, 1)
        total = len(weights)
    # small epsilon stops float error turning e.g. 2.9999999 into 2
    return {s: math.floor(hours * w / total + 1e-9) for s, w in weights.items()}
