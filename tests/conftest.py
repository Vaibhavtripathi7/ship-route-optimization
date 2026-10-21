import numpy as np
import pytest

from ship_routing.weather import WeatherField


@pytest.fixture
def make_field():
    """Calm, windless 0.5 degree grid with a constant wave height, optionally
    raised inside a box given as (min_lat, max_lat, min_lon, max_lon)."""

    def make(wave_m=1.0, size_deg=20.0, box=None, box_wave_m=15.0, hours=240):
        lats = np.arange(0.0, size_deg + 0.25, 0.5)
        lons = np.arange(0.0, size_deg + 0.25, 0.5)
        shape = (hours // 3 + 1, len(lats), len(lons))
        waves = np.full(shape, wave_m, dtype=np.float32)
        if box:
            min_lat, max_lat, min_lon, max_lon = box
            rows = (lats >= min_lat) & (lats <= max_lat)
            cols = (lons >= min_lon) & (lons <= max_lon)
            waves[:, rows[:, None] & cols[None, :]] = box_wave_m
        wind = np.zeros(shape, dtype=np.float32)
        return WeatherField(lats, lons, 3, wind, wind, waves)

    return make
