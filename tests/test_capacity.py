import pytest

from src.capacity import Capacity


def build(tutors, demand):
    capacity = Capacity()
    capacity.update_capacity(tutors, demand)
    return capacity


# ---------- update_capacity ----------


def test_single_subject_tutors_count_in_full():
    capacity = build([(10, ["Maths"], 1), (4, ["maths"], 1), (6, ["Maths"], 0)], demand={})
    assert capacity.get_subject_capacity("Maths", 1) == 14
    assert capacity.get_subject_capacity("Maths", 0) == 6


def test_multi_subject_hours_follow_demand():
    # two maths+english tutors, demand 16 maths : 2 english -> each splits 8:1 after rounding down
    tutors = [(10, ["maths", "english"], 0), (10, ["maths", "english"], 0)]
    capacity = build(tutors, {"maths": {0: 16}, "english": {0: 2}})
    assert capacity.get_subject_capacity("maths", 0) == 16
    assert capacity.get_subject_capacity("english", 0) == 2


def test_split_uses_demand_left_after_single_subject_tutors():
    # single maths tutor covers 12 of 16 maths -> unmet maths 4, english 8
    # maths+english tutor splits 10 hours 4:8 -> 3 maths, 6 english (rounded down)
    tutors = [(12, ["maths"], 1), (10, ["maths", "english"], 1)]
    capacity = build(tutors, {"maths": {1: 16}, "english": {1: 8}})
    assert capacity.get_subject_capacity("maths", 1) == 15
    assert capacity.get_subject_capacity("english", 1) == 6


def test_genders_use_their_own_demand():
    tutors = [(10, ["maths", "english"], 0), (10, ["maths", "english"], 1)]
    capacity = build(tutors, {"maths": {0: 10, 1: 0}, "english": {0: 0, 1: 10}})
    assert capacity.get_subject_capacity("maths", 0) == 10
    assert capacity.get_subject_capacity("english", 1) == 10
    assert capacity.get_subject_capacity("english", 0) == 0


def test_no_unmet_demand_splits_equally():
    tutors = [(20, ["maths"], 0), (10, ["maths", "english"], 0)]
    capacity = build(tutors, {"maths": {0: 5}})  # maths covered, english has no demand
    assert capacity.get_subject_capacity("maths", 0) == 25
    assert capacity.get_subject_capacity("english", 0) == 5


def test_row_order_does_not_change_result():
    tutors = [(12, ["maths"], 1), (10, ["maths", "english"], 1), (9, ["english", "physics"], 1)]
    demand = {"maths": {1: 16}, "english": {1: 8}, "physics": {1: 4}}
    forwards = build(tutors, demand).capacity
    backwards = build(list(reversed(tutors)), demand).capacity
    assert forwards == backwards


def test_demand_subjects_are_case_insensitive():
    capacity = build([(10, ["maths", "english"], 0)], {"Maths": {0: 10}})
    assert capacity.get_subject_capacity("maths", 0) == 10


def test_update_capacity_replaces_previous_snapshot():
    capacity = build([(10, ["Maths"], 1)], {})
    capacity.update_capacity([(2, ["Maths"], 1)], {})
    assert capacity.get_subject_capacity("Maths", 1) == 2


def test_invalid_row_rejected():
    with pytest.raises(ValueError, match="gender"):
        build([(10, ["maths"], 2)], {})


# ---------- add / use / get ----------


def test_unknown_subject_has_zero_capacity():
    assert Capacity().get_subject_capacity("physics", 0) == 0


def test_add_hours_for_new_recruit():
    capacity = Capacity()
    capacity.add_hours("Maths", 1, 8)
    assert capacity.get_subject_capacity("maths", 1) == 8
    assert capacity.get_subject_capacity("maths", 0) == 0


def test_add_hours_rejects_bad_gender():
    with pytest.raises(ValueError, match="gender"):
        Capacity().add_hours("maths", 2, 8)


def test_use_hours_reduces_subject_hours():
    capacity = build([(10, ["maths"], 0)], {})
    capacity.use_hours("Maths", 0, 3)
    assert capacity.get_subject_capacity("maths", 0) == 7


def test_use_hours_cannot_exceed_available():
    capacity = build([(4, ["maths"], 0)], {})
    with pytest.raises(ValueError, match="only 4 available"):
        capacity.use_hours("maths", 0, 5)
    assert capacity.get_subject_capacity("maths", 0) == 4
