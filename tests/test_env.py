import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ship_routing import geo
from ship_routing.env import (
    HOLD,
    PORT,
    PROGRESS_VALUE,
    REWARD_SCALE,
    START_SPEED,
    ShipRoutingEnv,
    random_voyage,
)
from ship_routing.weather import Storm, WeatherField


def run(env, actions=None, **options):
    obs, info = env.reset(options=options)
    rewards, terminated, step = [], False, 0
    while not terminated:
        action = actions[step] if actions and step < len(actions) else HOLD
        obs, reward, terminated, truncated, info = env.step(action)
        assert not truncated
        rewards.append(reward)
        step += 1
    return rewards, info


def test_passes_gymnasium_checks():
    check_env(ShipRoutingEnv([WeatherField.synthetic(0)]), skip_render_check=True)


def test_seeded_reset_is_reproducible():
    env = ShipRoutingEnv([WeatherField.synthetic(0), WeatherField.synthetic(1)])
    first, _ = env.reset(seed=7)
    start, field = env.position, env.field
    second, _ = env.reset(seed=7)
    assert np.array_equal(first, second)
    assert env.position == start and env.field is field


def test_holding_course_arrives(make_field):
    env = ShipRoutingEnv([make_field()])
    start, goal = (10.0, 2.0), (10.0, 7.0)
    rewards, info = run(env, start=start, goal=goal, departure_hour=0.0)
    assert info["outcome"] == "arrived"
    distance = geo.distance_nm(*start, *goal)
    assert info["hours"] == math.ceil((distance - 10) / START_SPEED)
    assert info["fuel_mt"] > 0
    assert sum(rewards) > 0


def test_leaving_the_area_forfeits_the_remaining_miles(make_field):
    env = ShipRoutingEnv([make_field()])
    turn_around = [PORT] * 12
    rewards, info = run(
        env, turn_around, start=(18.0, 10.0), goal=(14.0, 10.0), departure_hour=0.0
    )
    assert info["outcome"] == "left_area"
    assert rewards[-1] < -PROGRESS_VALUE * 4 * 60 / REWARD_SCALE


def test_forecast_running_out_ends_the_voyage(make_field):
    env = ShipRoutingEnv([make_field(hours=24)])
    _, info = run(env, start=(1.0, 1.0), goal=(19.0, 19.0), departure_hour=10.0)
    assert info["outcome"] == "timeout"
    assert info["hours"] == 14


def test_storm_on_the_route_costs_fuel():
    # The first version added its storm to a time slice the ship never read.
    start, goal = (10.0, 60.0), (10.0, 70.0)
    storm = Storm(lat=10.0, lon=65.0)
    fuel = {}
    for name, storms in (("calm", ()), ("storm", [storm])):
        env = ShipRoutingEnv([WeatherField.synthetic(0, storms=storms)])
        _, info = run(env, start=start, goal=goal, departure_hour=0.0)
        fuel[name] = info["fuel_mt"]
    assert fuel["storm"] > 1.1 * fuel["calm"]


@pytest.mark.parametrize(
    "options",
    [
        {"start": (50.0, 5.0), "goal": (5.0, 5.0)},
        {"start": (5.0, 5.0), "goal": (6.0, 6.0), "departure_hour": 240.0},
    ],
)
def test_rejects_bad_options(make_field, options):
    with pytest.raises(ValueError):
        ShipRoutingEnv([make_field()]).reset(options=options)


def test_random_voyage_needs_room():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_voyage(rng, (0.0, 3.0, 0.0, 3.0))
