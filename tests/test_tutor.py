import pytest

from src.tutor import clean_tutor_row, split_hours

# ---------- clean_tutor_row ----------


def test_clean_row_lowercases_and_deduplicates_subjects():
    assert clean_tutor_row(5, [" Maths", "maths", "English"], 0) == (5, ["maths", "english"], 0)


@pytest.mark.parametrize(
    "hours, subjects, gender, message",
    [
        (10, ["maths"], 2, "gender"),
        (10, ["maths"], "1", "gender"),
        (-1, ["maths"], 0, "hours"),
        (2.5, ["maths"], 0, "hours"),
        (10, [], 0, "at least one subject"),
    ],
)
def test_invalid_row_rejected(hours, subjects, gender, message):
    with pytest.raises(ValueError, match=message):
        clean_tutor_row(hours, subjects, gender)


# ---------- split_hours ----------


def test_single_subject_gets_all_hours():
    assert split_hours(7, {"maths": 3}) == {"maths": 7}


def test_split_is_proportional():
    assert split_hours(10, {"maths": 3, "english": 2}) == {"maths": 6, "english": 4}


def test_split_rounds_down():
    # 7 * 3/5 = 4.2 -> 4, 7 * 2/5 = 2.8 -> 2; one hour unallocated
    split = split_hours(7, {"maths": 3, "english": 2})
    assert split == {"maths": 4, "english": 2}
    assert all(isinstance(h, int) for h in split.values())


def test_split_exact_thirds_not_lost_to_float_error():
    assert split_hours(9, {"a": 0.1, "b": 0.1, "c": 0.1}) == {"a": 3, "b": 3, "c": 3}


def test_all_zero_weights_split_equally():
    assert split_hours(10, {"maths": 0, "english": 0}) == {"maths": 5, "english": 5}


def test_zero_weight_subject_gets_nothing():
    assert split_hours(10, {"maths": 4, "english": 0}) == {"maths": 10, "english": 0}
