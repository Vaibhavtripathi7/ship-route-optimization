import pytest

from ship_routing.physics import KNOT, ShipPhysics, voyage_cost
from ship_routing.weather import CALM, Conditions

ship = ShipPhysics()


def test_calm_water_resistance_matches_ittc57_by_hand():
    # 15 kn = 7.717 m/s, Re = 7.717 * 200 / 1.19e-6 = 1.297e9
    # Cf = 0.075 / (log10(Re) - 2)^2 = 0.0014824
    # R = 0.5 * 1025 * 9430 m^2 * 1.3 * Cf * 7.717^2 = 554.6 kN
    assert ship.calm_water_resistance(15) == pytest.approx(554.6e3, rel=1e-3)


def test_wave_resistance_matches_stawave1_by_hand():
    # 1025 * 9.81 * 2^2 * 32 * sqrt(32 / 200) / 16 = 32.18 kN
    assert ship.wave_resistance(2.0) == pytest.approx(32.18e3, rel=1e-3)


def test_wind_depends_on_direction():
    # Heading north, so v < 0 blows from ahead and v > 0 from astern.
    headwind = ship.fuel_rate(12, 0.0, Conditions(0.0, -10.0, 0.0))
    beam_wind = ship.fuel_rate(12, 0.0, Conditions(10.0, 0.0, 0.0))
    calm = ship.fuel_rate(12, 0.0, CALM)
    tailwind = ship.fuel_rate(12, 0.0, Conditions(0.0, 5.0, 0.0))
    assert headwind > calm > tailwind
    assert beam_wind == pytest.approx(calm)


def test_strong_tailwind_never_pushes_the_ship():
    air_moving_with_ship = Conditions(0.0, 12 * KNOT, 0.0)
    gale_astern = Conditions(0.0, 40.0, 0.0)
    assert ship.fuel_rate(12, 0.0, gale_astern) == pytest.approx(
        ship.fuel_rate(12, 0.0, air_moving_with_ship)
    )


def test_fuel_rate_rises_with_speed_and_is_zero_at_rest():
    rates = [ship.fuel_rate(v, 0.0, CALM) for v in range(0, 19)]
    assert rates[0] == 0.0
    assert all(a < b for a, b in zip(rates, rates[1:]))
    # A Panamax bulker burns roughly 20 to 30 t a day at 14 kn.
    assert 20 < 24 * ship.fuel_rate(14, 0.0, CALM) < 30


def test_voyage_cost_prices_fuel_and_time():
    assert voyage_cost(10.0, 24.0) == 10 * 600 + 24 * 500
