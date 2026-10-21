import pytest
from fastapi.testclient import TestClient

from ship_routing.service import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("planner", ["ppo", "astar", "direct"])
def test_route(client, planner):
    body = {"start": [10.0, 60.0], "goal": [15.0, 70.0], "planner": planner}
    response = client.post("/route", json=body)
    assert response.status_code == 200
    route = response.json()
    assert route["outcome"] == "arrived"
    assert route["track"][0] == [10.0, 60.0]
    assert route["cost_usd"] > 0


@pytest.mark.parametrize(
    "body",
    [
        {"start": [60.0, 60.0], "goal": [15.0, 70.0]},
        {"start": [10.0, 60.0], "goal": [15.0, 70.0], "departure_hour": 500},
        {"start": [10.0, 60.0], "goal": [15.0, 70.0], "planner": "dijkstra"},
        {"start": [10.0, 60.0], "goal": [15.0, 70.0], "speed_knots": 30},
    ],
)
def test_rejects_bad_requests(client, body):
    assert client.post("/route", json=body).status_code == 422
