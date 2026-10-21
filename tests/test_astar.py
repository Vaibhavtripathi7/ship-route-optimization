import pytest

from ship_routing import geo
from ship_routing.astar import plan_route
from ship_routing.physics import ShipPhysics

ship = ShipPhysics()


def route_length(route):
    return sum(geo.distance_nm(*a, *b) for a, b in zip(route, route[1:]))


def test_route_runs_from_start_to_goal_in_short_legs(make_field):
    start, goal = (1.2, 0.7), (17.9, 13.3)
    route = plan_route(make_field(), ship, start, goal, 12.0)
    assert route[0] == start and route[-1] == goal
    # Knight moves are the longest legs: about 67 nm on a 0.5 degree grid.
    assert max(geo.distance_nm(*a, *b) for a, b in zip(route, route[1:])) < 70


def test_route_is_nearly_straight_in_uniform_sea(make_field):
    start, goal = (2.0, 1.0), (12.0, 8.0)
    route = plan_route(make_field(), ship, start, goal, 12.0)
    assert route_length(route) < 1.03 * geo.distance_nm(*start, *goal)


def test_route_goes_around_high_waves(make_field):
    field = make_field(box=(8.0, 12.0, 9.0, 11.0))
    route = plan_route(field, ship, (10.0, 2.0), (10.0, 18.0), 12.0)
    assert all(field.at(*point, 0).wave_height < 15 for point in route)


def test_start_and_goal_in_the_same_cell(make_field):
    assert plan_route(make_field(), ship, (5.0, 5.0), (5.1, 5.1), 12.0) == [
        (5.0, 5.0),
        (5.1, 5.1),
    ]


@pytest.mark.parametrize(
    "start, goal, speed",
    [((5.0, 5.0), (50.0, 5.0), 12.0), ((5.0, 5.0), (6.0, 6.0), 0.0)],
)
def test_rejects_bad_requests(make_field, start, goal, speed):
    with pytest.raises(ValueError):
        plan_route(make_field(), ship, start, goal, speed)
