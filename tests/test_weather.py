import numpy as np
import pytest

from ship_routing.weather import Storm, WeatherField


def test_same_seed_gives_same_forecast():
    a, b = WeatherField.synthetic(3), WeatherField.synthetic(3)
    assert np.array_equal(a.wave_height, b.wave_height)
    assert not np.array_equal(a.wave_height, WeatherField.synthetic(4).wave_height)


def test_weather_is_spatially_smooth():
    waves = WeatherField.synthetic(0, storms=()).wave_height
    left, right = waves[:, :, :-1].ravel(), waves[:, :, 1:].ravel()
    assert np.corrcoef(left, right)[0, 1] > 0.9


def test_storm_moves_with_its_drift():
    storm = Storm(lat=10.0, lon=70.0, drift_lon=0.1)
    calm = WeatherField.synthetic(0, storms=())
    stormy = WeatherField.synthetic(0, storms=[storm])

    def storm_waves(lat, lon, hour):
        return (
            stormy.at(lat, lon, hour).wave_height - calm.at(lat, lon, hour).wave_height
        )

    assert storm_waves(10, 70, 0) == pytest.approx(storm.peak_wave_m, rel=1e-3)
    assert storm_waves(10, 76, 60) == pytest.approx(storm.peak_wave_m, rel=1e-3)
    assert storm_waves(10, 70, 60) < 1.0


def test_lookup_clamps_to_grid_edge_and_horizon():
    field = WeatherField.synthetic(0)
    min_lat, _, min_lon, _ = field.bounds
    assert not field.contains(-50.0, 0.0)
    assert field.at(-50.0, 0.0, -5.0) == field.at(min_lat, min_lon, 0.0)
    assert field.at(10.0, 70.0, 1e6) == field.at(10.0, 70.0, field.horizon_hours)


def test_rejects_uneven_grid():
    lats = np.array([0.0, 0.5, 1.5])
    lons = np.array([0.0, 0.5])
    data = np.zeros((2, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        WeatherField(lats, lons, 3, data, data, data)
