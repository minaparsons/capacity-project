from src.demand import Demand


def test_get_subject_demand_is_case_insensitive():
    demand = Demand({"maths": {0: 8, 1: 12}})
    assert demand.get_subject_demand("Maths", 0) == 8
    assert demand.get_subject_demand("MATHS", 1) == 12
