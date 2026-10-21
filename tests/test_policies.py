from pathlib import Path

import pytest

from ship_routing import geo
from ship_routing.env import ARRIVAL_NM, ShipRoutingEnv
from ship_routing.evaluate import scenarios
from ship_routing.policies import AStarPolicy, DirectPolicy, PPOPolicy, sail

MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "ppo_seed0.zip"


def test_direct_policy_track_starts_at_the_start(make_field):
    # The first version plotted a track whose first point aliased the final one.
    env = ShipRoutingEnv([make_field()])
    start, goal = (3.0, 3.0), (15.0, 12.0)
    voyage = sail(env, DirectPolicy(12.0), start, goal)
    assert voyage.outcome == "arrived"
    assert voyage.track[0] == start
    assert geo.distance_nm(*voyage.track[-1], *goal) <= ARRIVAL_NM + 12
    assert len(voyage.track) == voyage.hours + 1


def test_direct_policy_holds_its_speed(make_field):
    env = ShipRoutingEnv([make_field()])
    voyage = sail(env, DirectPolicy(16.0), (3.0, 3.0), (3.0, 15.0))
    legs = [geo.distance_nm(*a, *b) for a, b in zip(voyage.track, voyage.track[1:])]
    assert legs[-1] == pytest.approx(16.0, rel=1e-3)


@pytest.mark.parametrize("name", ["open sea 45", "crossing 31"])
def test_astar_follower_does_not_circle_missed_waypoints(name):
    scenario = next(s for s in scenarios() if s.name == name)
    env = ShipRoutingEnv([scenario.field()])
    voyage = sail(
        env, AStarPolicy(11.0), scenario.start, scenario.goal, scenario.departure_hour
    )
    assert voyage.outcome == "arrived"


def test_trained_policy_reaches_the_showcase_goal():
    scenario = next(s for s in scenarios() if s.suite == "showcase")
    env = ShipRoutingEnv([scenario.field()])
    voyage = sail(
        env, PPOPolicy(MODEL), scenario.start, scenario.goal, scenario.departure_hour
    )
    assert voyage.outcome == "arrived"
