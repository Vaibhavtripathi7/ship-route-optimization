import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ship_routing import geo
from ship_routing.physics import ShipPhysics, voyage_cost

HOLD, PORT, STARBOARD, FASTER, SLOWER = range(5)

DT_HOURS = 1.0
TURN_DEG = 15.0
MIN_SPEED, MAX_SPEED = 8.0, 18.0
START_SPEED = 12.0
ARRIVAL_NM = 10.0
# Credit per mile made good. It sits above the cost of a mile at any sensible
# speed, so the agent always wants to arrive, and it is shaped as a potential
# difference, so it does not change which arrival is cheapest.
PROGRESS_VALUE = 150.0
REWARD_SCALE = 1e4
# Forecast probes as (bearing offset from heading, range in nm).
LOOKAHEAD = [(offset, rng) for rng in (50, 150) for offset in (-45, 0, 45)]


def random_voyage(rng, bounds, min_nm=300.0, max_nm=900.0, margin_deg=1.0):
    """Uniform start and goal inside bounds, a sensible distance apart."""
    min_lat, max_lat, min_lon, max_lon = bounds
    for _ in range(1000):
        start = (
            float(rng.uniform(min_lat + margin_deg, max_lat - margin_deg)),
            float(rng.uniform(min_lon + margin_deg, max_lon - margin_deg)),
        )
        goal = (
            float(rng.uniform(min_lat + margin_deg, max_lat - margin_deg)),
            float(rng.uniform(min_lon + margin_deg, max_lon - margin_deg)),
        )
        if min_nm <= geo.distance_nm(*start, *goal) <= max_nm:
            return start, goal
    raise ValueError("bounds are too small for voyages of the requested length")


class ShipRoutingEnv(gym.Env):
    """One voyage through a forecast, decided hour by hour.

    Actions: hold, turn port or starboard by TURN_DEG, or change speed by one
    knot. Reward per hour is progress credit minus fuel and charter cost.
    The episode ends on arrival, on leaving the forecast area, or when the
    forecast runs out; the last two forfeit the credit for the miles left.

    reset options: start, goal, departure_hour. Missing values are drawn at
    random, which is how training sees varied voyages and weather.
    """

    metadata = {"render_modes": []}

    def __init__(self, fields, physics=None):
        if not fields:
            raise ValueError("need at least one weather field")
        self.fields = list(fields)
        self.physics = physics or ShipPhysics()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(6 + len(LOOKAHEAD),), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.field = self.fields[self.np_random.integers(len(self.fields))]
        start, goal = options.get("start"), options.get("goal")
        if start is None or goal is None:
            start, goal = random_voyage(self.np_random, self.field.bounds)
        if not (self.field.contains(*start) and self.field.contains(*goal)):
            raise ValueError("start and goal must lie inside the weather field")
        departure = options.get("departure_hour")
        if departure is None:
            departure = float(self.np_random.uniform(0, self.field.horizon_hours / 2))
        if not 0 <= departure < self.field.horizon_hours:
            raise ValueError("departure must fall inside the forecast horizon")

        self.position = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.hour = float(departure)
        self.departure_hour = self.hour
        self.heading = self.bearing_to_goal()
        self.speed = START_SPEED
        self.remaining_nm = geo.distance_nm(*self.position, *self.goal)
        self.fuel_mt = 0.0
        self.sailed_nm = 0.0
        return self._obs(), self._info(None)

    def step(self, action):
        if action == PORT:
            self.heading = (self.heading - TURN_DEG) % 360
        elif action == STARBOARD:
            self.heading = (self.heading + TURN_DEG) % 360
        elif action == FASTER:
            self.speed = min(self.speed + 1, MAX_SPEED)
        elif action == SLOWER:
            self.speed = max(self.speed - 1, MIN_SPEED)

        leg = self.speed * DT_HOURS
        previous = self.position
        midpoint = geo.displace(*previous, self.heading, leg / 2)
        weather = self.field.at(*midpoint, self.hour + DT_HOURS / 2)
        fuel = self.physics.fuel_rate(self.speed, self.heading, weather) * DT_HOURS
        self.position = geo.displace(*previous, self.heading, leg)
        self.hour += DT_HOURS
        self.fuel_mt += fuel
        self.sailed_nm += leg

        arrived = (
            geo.closest_approach_nm(previous, self.position, self.goal) <= ARRIVAL_NM
        )
        remaining = 0.0 if arrived else geo.distance_nm(*self.position, *self.goal)
        cost = voyage_cost(fuel, DT_HOURS)
        reward = PROGRESS_VALUE * (self.remaining_nm - remaining) - cost
        self.remaining_nm = remaining

        if arrived:
            outcome = "arrived"
        elif not self.field.contains(*self.position):
            outcome = "left_area"
        elif self.hour >= self.field.horizon_hours:
            outcome = "timeout"
        else:
            outcome = None
        if outcome in ("left_area", "timeout"):
            reward -= PROGRESS_VALUE * remaining

        obs = self._obs()
        return (
            obs,
            reward / REWARD_SCALE,
            outcome is not None,
            False,
            self._info(outcome),
        )

    def bearing_to_goal(self):
        return geo.bearing_deg(*self.position, *self.goal)

    def _obs(self):
        lat, lon = self.position
        error = math.radians(geo.angle_diff(self.bearing_to_goal(), self.heading))
        here = self.field.at(lat, lon, self.hour)
        heading = math.radians(self.heading)
        headwind = -(here.u_wind * math.sin(heading) + here.v_wind * math.cos(heading))
        ahead = []
        for offset, distance in LOOKAHEAD:
            plat, plon = geo.displace(lat, lon, self.heading + offset, distance)
            eta = self.hour + distance / self.speed
            ahead.append(self.field.at(plat, plon, eta).wave_height / 10)
        return np.array(
            [
                self.remaining_nm / 1000,
                math.sin(error),
                math.cos(error),
                (self.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED),
                headwind / 20,
                here.wave_height / 10,
                *ahead,
            ],
            dtype=np.float32,
        )

    def _info(self, outcome):
        hours = self.hour - self.departure_hour
        return {
            "outcome": outcome,
            "fuel_mt": self.fuel_mt,
            "hours": hours,
            "sailed_nm": self.sailed_nm,
            "cost_usd": voyage_cost(self.fuel_mt, hours),
        }
