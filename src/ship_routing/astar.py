import heapq
import itertools
import math

from ship_routing import geo
from ship_routing.physics import voyage_cost
from ship_routing.weather import CALM

# 16 connected grid: the 8 neighbours plus knight moves, so legs can run at
# about 27 degree steps instead of only multiples of 45.
MOVES = [
    (di, dj)
    for di in range(-2, 3)
    for dj in range(-2, 3)
    if math.gcd(abs(di), abs(dj)) == 1
]


def plan_route(field, physics, start, goal, speed_knots, departure_hour=0.0):
    """Cheapest grid route at constant speed through a time varying forecast.

    Each leg is costed with the forecast at the leg midpoint and at the time
    the ship gets there. Returns waypoints from start to goal inclusive.
    """
    if speed_knots <= 0:
        raise ValueError("speed must be positive")
    if not (field.contains(*start) and field.contains(*goal)):
        raise ValueError("start and goal must lie inside the weather field")

    def centre(cell):
        return float(field.lats[cell[0]]), float(field.lons[cell[1]])

    # Calm water fuel plus charter hire is a lower bound on the cost of a mile.
    cost_per_nm = (
        voyage_cost(physics.fuel_rate(speed_knots, 0.0, CALM), 1.0) / speed_knots
    )
    goal_lat, goal_lon = centre(field.cell(*goal))

    def heuristic(cell):
        return cost_per_nm * geo.distance_nm(*centre(cell), goal_lat, goal_lon)

    source, target = field.cell(*start), field.cell(*goal)
    n_lat, n_lon = len(field.lats), len(field.lons)
    cost = {source: 0.0}
    hour = {source: departure_hour}
    parent = {}
    done = set()
    tie = itertools.count()
    frontier = [(heuristic(source), next(tie), source)]

    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell == target:
            break
        if cell in done:
            continue
        done.add(cell)
        lat, lon = centre(cell)
        for di, dj in MOVES:
            nxt = (cell[0] + di, cell[1] + dj)
            if not (0 <= nxt[0] < n_lat and 0 <= nxt[1] < n_lon) or nxt in done:
                continue
            nlat, nlon = centre(nxt)
            dist = geo.distance_nm(lat, lon, nlat, nlon)
            hours = dist / speed_knots
            weather = field.at(
                (lat + nlat) / 2, (lon + nlon) / 2, hour[cell] + hours / 2
            )
            heading = geo.bearing_deg(lat, lon, nlat, nlon)
            fuel = physics.fuel_rate(speed_knots, heading, weather) * hours
            new_cost = cost[cell] + voyage_cost(fuel, hours)
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt] = new_cost
                hour[nxt] = hour[cell] + hours
                parent[nxt] = cell
                heapq.heappush(frontier, (new_cost + heuristic(nxt), next(tie), nxt))

    cells = []
    cell = target
    while cell != source:
        cells.append(cell)
        cell = parent[cell]
    inner = [centre(c) for c in reversed(cells[1:])]
    return [tuple(start), *inner, tuple(goal)]
