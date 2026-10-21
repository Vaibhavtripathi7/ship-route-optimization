from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

INDIAN_OCEAN = (-10.0, 30.0, 50.0, 100.0)


@dataclass(frozen=True)
class Conditions:
    u_wind: float  # m/s, positive towards east
    v_wind: float  # m/s, positive towards north
    wave_height: float  # significant wave height, m


CALM = Conditions(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Storm:
    """A cyclonic storm whose centre drifts at constant velocity."""

    lat: float
    lon: float
    radius_deg: float = 2.5
    peak_wave_m: float = 7.0
    peak_wind_ms: float = 20.0
    drift_lat: float = 0.0  # degrees per hour
    drift_lon: float = 0.0

    def centre(self, hour):
        return self.lat + self.drift_lat * hour, self.lon + self.drift_lon * hour


class WeatherField:
    """Gridded forecast of wind and waves, indexed by (time, lat, lon)."""

    def __init__(self, lats, lons, step_hours, u_wind, v_wind, wave_height):
        shape = (len(u_wind), len(lats), len(lons))
        if not (u_wind.shape == v_wind.shape == wave_height.shape == shape):
            raise ValueError("weather arrays must have shape (time, lat, lon)")
        if len(lats) < 2 or len(lons) < 2:
            raise ValueError("grid needs at least two points per axis")
        resolution = float(lats[1] - lats[0])
        if not (
            np.allclose(np.diff(lats), resolution)
            and np.allclose(np.diff(lons), resolution)
        ):
            raise ValueError("lats and lons must share one even spacing")
        self.lats, self.lons = lats, lons
        self.resolution = resolution
        self.bounds = (float(lats[0]), float(lats[-1]), float(lons[0]), float(lons[-1]))
        self.step_hours = step_hours
        self.horizon_hours = (shape[0] - 1) * step_hours
        self.u_wind, self.v_wind, self.wave_height = u_wind, v_wind, wave_height

    @classmethod
    def synthetic(
        cls,
        seed,
        storms=None,
        bounds=INDIAN_OCEAN,
        resolution=0.5,
        horizon_hours=240,
        step_hours=3,
    ):
        """Smooth random wind and swell plus moving storms.

        storms=None draws zero to three random storms from the seed; pass a
        sequence (possibly empty) to place them explicitly.
        """
        rng = np.random.default_rng(seed)
        min_lat, max_lat, min_lon, max_lon = bounds
        lats = np.arange(min_lat, max_lat + resolution / 2, resolution)
        lons = np.arange(min_lon, max_lon + resolution / 2, resolution)
        hours = np.arange(0, horizon_hours + step_hours / 2, step_hours)
        shape = (len(hours), len(lats), len(lons))

        def smooth_noise():
            sigma = (12 / step_hours, 2 / resolution, 2 / resolution)
            field = gaussian_filter(rng.standard_normal(shape), sigma)
            return field / field.std()

        if storms is None:
            storms = [_random_storm(rng, bounds) for _ in range(rng.integers(0, 4))]

        band = np.sin(np.pi * (lats - min_lat) / (max_lat - min_lat))[None, :, None]
        u_wind = 5 * band + 4 * smooth_noise()
        v_wind = 4 * smooth_noise()
        swell = np.clip(1.0 + 0.4 * smooth_noise(), 0.2, None)
        # Pierson-Moskowitz fully developed sea: Hs = 0.21 U^2 / g
        wave_height = swell + 0.0214 * (u_wind**2 + v_wind**2)

        lat_grid = lats[None, :, None]
        lon_grid = lons[None, None, :]
        for storm in storms:
            c_lat, c_lon = storm.centre(hours[:, None, None])
            dy = lat_grid - c_lat
            dx = (lon_grid - c_lon) * np.cos(np.radians(c_lat))
            r = np.hypot(dx, dy) / storm.radius_deg
            wave_height = wave_height + storm.peak_wave_m * np.exp(-(r**2) / 2)
            # Tangential wind peaking at the radius, anticlockwise in the north.
            speed = storm.peak_wind_ms * r * np.exp((1 - r**2) / 2)
            spin = np.where(c_lat >= 0, 1.0, -1.0)
            dist = np.maximum(np.hypot(dx, dy), 1e-9)
            u_wind = u_wind - spin * speed * dy / dist
            v_wind = v_wind + spin * speed * dx / dist

        def f32(a):
            return np.ascontiguousarray(a, dtype=np.float32)

        return cls(lats, lons, step_hours, f32(u_wind), f32(v_wind), f32(wave_height))

    def contains(self, lat, lon):
        min_lat, max_lat, min_lon, max_lon = self.bounds
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def cell(self, lat, lon):
        """Nearest grid indices, clamped to the grid edge."""
        i = round((lat - self.bounds[0]) / self.resolution)
        j = round((lon - self.bounds[2]) / self.resolution)
        return min(max(i, 0), len(self.lats) - 1), min(max(j, 0), len(self.lons) - 1)

    def at(self, lat, lon, hour):
        """Nearest neighbour lookup; times past the horizon reuse the last frame."""
        t = min(max(round(hour / self.step_hours), 0), len(self.u_wind) - 1)
        i, j = self.cell(lat, lon)
        return Conditions(
            float(self.u_wind[t, i, j]),
            float(self.v_wind[t, i, j]),
            float(self.wave_height[t, i, j]),
        )


def _random_storm(rng, bounds):
    min_lat, max_lat, min_lon, max_lon = bounds
    direction = rng.uniform(0, 2 * np.pi)
    drift = rng.uniform(0.05, 0.2)
    return Storm(
        lat=rng.uniform(min_lat, max_lat),
        lon=rng.uniform(min_lon, max_lon),
        radius_deg=rng.uniform(1.5, 3.5),
        peak_wave_m=rng.uniform(4.0, 9.0),
        peak_wind_ms=rng.uniform(12.0, 25.0),
        drift_lat=drift * np.cos(direction),
        drift_lon=drift * np.sin(direction),
    )
