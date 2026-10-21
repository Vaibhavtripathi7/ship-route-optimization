"""HTTP routing service. Run from the repository root:

    python -m ship_routing.service

Routes are simulated against a synthetic forecast chosen by
SHIP_ROUTING_WEATHER_SEED; SHIP_ROUTING_MODEL points at the PPO weights.
"""

import dataclasses
import os
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ship_routing.env import MAX_SPEED, MIN_SPEED, ShipRoutingEnv
from ship_routing.physics import ShipPhysics
from ship_routing.policies import AStarPolicy, DirectPolicy, PPOPolicy, sail
from ship_routing.weather import WeatherField

MODEL_PATH = os.environ.get("SHIP_ROUTING_MODEL", "artifacts/ppo_seed0.zip")
WEATHER_SEED = int(os.environ.get("SHIP_ROUTING_WEATHER_SEED", "20000"))

state = {}


@asynccontextmanager
async def lifespan(app):
    state["field"] = WeatherField.synthetic(WEATHER_SEED)
    state["physics"] = ShipPhysics()
    state["ppo"] = PPOPolicy(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Ship routing", lifespan=lifespan)


class RouteRequest(BaseModel):
    start: tuple[float, float]
    goal: tuple[float, float]
    departure_hour: float = Field(0.0, ge=0)
    planner: Literal["ppo", "astar", "direct"] = "ppo"
    # PPO picks its own speed; this only applies to astar and direct.
    speed_knots: float = Field(11.0, ge=MIN_SPEED, le=MAX_SPEED)


class RouteResponse(BaseModel):
    policy: str
    outcome: str
    fuel_mt: float
    hours: float
    cost_usd: float
    sailed_nm: float
    decision_ms: float
    track: list[tuple[float, float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route")
def route(request: RouteRequest) -> RouteResponse:
    if request.planner == "ppo":
        policy = state["ppo"]
    elif request.planner == "astar":
        policy = AStarPolicy(request.speed_knots)
    else:
        policy = DirectPolicy(request.speed_knots)
    env = ShipRoutingEnv([state["field"]], state["physics"])
    try:
        voyage = sail(env, policy, request.start, request.goal, request.departure_hour)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RouteResponse(**dataclasses.asdict(voyage))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
