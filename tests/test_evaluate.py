from ship_routing.evaluate import BASELINE, Scenario, summary_table
from ship_routing.policies import Voyage


def voyage(policy, cost, outcome="arrived"):
    return Voyage(policy, outcome, 10.0, 20.0, cost, 200.0, 1.0, [])


def scenario(name):
    return Scenario(name, "test", 0, (0.0, 0.0), (1.0, 1.0), 0.0)


def table_rows(table):
    return {line.split(" | ")[0].strip("| "): line for line in table.splitlines()}


def test_cost_change_is_paired_against_the_baseline():
    results = [
        (scenario("a"), [voyage(BASELINE, 100.0), voyage("other", 110.0)]),
        (scenario("b"), [voyage(BASELINE, 200.0), voyage("other", 220.0)]),
    ]
    rows = table_rows(summary_table(results))
    assert "+10.00% ± 0.00" in rows["other"]
    assert "| 2/2 |" in rows["other"]


def test_voyages_some_policy_failed_are_left_out_of_the_means():
    results = [
        (scenario("a"), [voyage(BASELINE, 100.0), voyage("other", 90.0)]),
        (scenario("b"), [voyage(BASELINE, 100.0), voyage("other", 80.0)]),
        (scenario("c"), [voyage(BASELINE, 100.0), voyage("other", 5.0, "timeout")]),
    ]
    table = summary_table(results)
    rows = table_rows(table)
    assert table.startswith("Means over the 2 voyages")
    assert "| 2/3 |" in rows["other"]
    assert "-15.00%" in rows["other"]


def test_reports_when_nothing_is_comparable():
    results = [(scenario("a"), [voyage(BASELINE, 100.0, "left_area")])]
    assert summary_table(results) == "No voyage was completed by every policy."
