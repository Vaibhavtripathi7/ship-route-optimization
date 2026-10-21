import time
from dataclasses import dataclass

from stable_baselines3 import PPO

from ship_routing import geo
from ship_routing.astar import plan_route
from ship_routing.env import DT_HOURS, FASTER, HOLD, PORT, SLOWER, STARBOARD, TURN_DEG


def steer(env, bearing, speed):
    """Action that moves the ship towards a bearing, then towards a speed."""
    error = geo.angle_diff(bearing, env.heading)
    if error > TURN_DEG / 2:
        return STARBOARD
    if error < -TURN_DEG / 2:
        return PORT
    if env.speed < speed - 0.5:
        return FASTER
    if env.speed > speed + 0.5:
        return SLOWER
    return HOLD


class DirectPolicy:
    """Rhumb line to the destination at a fixed speed: no weather routing."""

    def __init__(self, speed_knots):
        self.speed = speed_knots
        self.name = f"direct {speed_knots:g} kn"

    def reset(self, env):
        pass

    def act(self, obs, env):
        return steer(env, env.bearing_to_goal(), self.speed)


class AStarPolicy:
    """Plan once on the forecast at departure, then follow the waypoints."""

    def __init__(self, speed_knots):
        self.speed = speed_knots
        self.name = f"A* {speed_knots:g} kn"

    def reset(self, env):
        route = plan_route(
            env.field, env.physics, env.position, env.goal, self.speed, env.hour
        )
        self.waypoints = route[1:]

    def act(self, obs, env):
        # Move on once a waypoint is within one hour's run, or once the ship is
        # already closer to the next one; a ship that turns 15 degrees an hour
        # would otherwise circle a waypoint it narrowly missed.
        dist = geo.distance_nm
        here = env.position
        while len(self.waypoints) > 1:
            current, following = self.waypoints[0], self.waypoints[1]
            reached = dist(*here, *current) <= self.speed * DT_HOURS
            passed = dist(*here, *following) <= dist(*current, *following)
            if not (reached or passed):
                break
            self.waypoints.pop(0)
        return steer(env, geo.bearing_deg(*here, *self.waypoints[0]), self.speed)


class PPOPolicy:
    def __init__(self, path, name="PPO"):
        self.model = PPO.load(path, device="cpu")
        self.name = name

    def reset(self, env):
        pass

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


@dataclass
class Voyage:
    policy: str
    outcome: str
    fuel_mt: float
    hours: float
    cost_usd: float
    sailed_nm: float
    decision_ms: float  # time spent planning and choosing actions
    track: list


def sail(env, policy, start, goal, departure_hour=0.0):
    obs, _ = env.reset(
        options={"start": start, "goal": goal, "departure_hour": departure_hour}
    )
    track = [env.position]
    clock = time.perf_counter()
    policy.reset(env)
    decision = time.perf_counter() - clock
    done = False
    while not done:
        clock = time.perf_counter()
        action = policy.act(obs, env)
        decision += time.perf_counter() - clock
        obs, _, terminated, truncated, info = env.step(action)
        track.append(env.position)
        done = terminated or truncated
    return Voyage(
        policy=policy.name,
        outcome=info["outcome"],
        fuel_mt=info["fuel_mt"],
        hours=info["hours"],
        cost_usd=info["cost_usd"],
        sailed_nm=info["sailed_nm"],
        decision_ms=decision * 1000,
        track=track,
    )
