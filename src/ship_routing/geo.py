"""Local flat earth navigation in nautical miles.

Voyages in this project are under a thousand miles at low latitude, where an
equirectangular projection around the leg's mid latitude is accurate to well
under one percent.
"""

import math

NM_PER_DEG = 60.0


def _local_xy(lat, lon, ref_lat, ref_lon):
    x = (lon - ref_lon) * NM_PER_DEG * math.cos(math.radians((lat + ref_lat) / 2))
    y = (lat - ref_lat) * NM_PER_DEG
    return x, y


def distance_nm(lat1, lon1, lat2, lon2):
    return math.hypot(*_local_xy(lat2, lon2, lat1, lon1))


def bearing_deg(lat1, lon1, lat2, lon2):
    """True bearing from point 1 to point 2, clockwise from north."""
    x, y = _local_xy(lat2, lon2, lat1, lon1)
    return math.degrees(math.atan2(x, y)) % 360


def displace(lat, lon, heading_deg, distance):
    heading = math.radians(heading_deg)
    dlat = distance * math.cos(heading) / NM_PER_DEG
    mid_lat = math.radians(lat + dlat / 2)
    dlon = distance * math.sin(heading) / (NM_PER_DEG * math.cos(mid_lat))
    return lat + dlat, lon + dlon


def angle_diff(a, b):
    """Signed difference a minus b, wrapped to [-180, 180)."""
    return (a - b + 180) % 360 - 180


def closest_approach_nm(start, end, point):
    """Smallest distance from point to the straight leg between start and end."""
    ax, ay = _local_xy(*start, *point)
    bx, by = _local_xy(*end, *point)
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else min(max(-(ax * dx + ay * dy) / length_sq, 0.0), 1.0)
    return math.hypot(ax + t * dx, ay + t * dy)
