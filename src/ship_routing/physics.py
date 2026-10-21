import math

KNOT = 0.514444  # m/s
GRAVITY = 9.81
RHO_WATER = 1025.0
RHO_AIR = 1.225
NU_WATER = 1.19e-6  # kinematic viscosity of sea water at 15 C, m^2/s

FUEL_PRICE = 600.0  # USD per tonne of VLSFO
HOURLY_COST = 500.0  # USD per hour of charter hire, about 12k a day for a Panamax


def voyage_cost(fuel_mt, hours):
    return fuel_mt * FUEL_PRICE + hours * HOURLY_COST


class ShipPhysics:
    """Resistance and fuel model of a Panamax size bulk carrier.

    Calm water: ITTC-57 friction line with a form factor.
    Waves: STAWAVE-1 added resistance (ISO 15016), applied as if every sea came
    from ahead, which overestimates beam and following seas.
    Wind: drag of the apparent wind along the ship axis. Wind from astern
    reduces it to zero but never pushes the ship.
    """

    def __init__(
        self,
        length=200.0,
        beam=32.0,
        draft=12.0,
        block_coeff=0.8,
        form_factor=0.3,
        windage_height=24.0,
        wind_drag_coeff=0.8,
        propulsive_efficiency=0.7,
        sfoc=180.0,  # specific fuel oil consumption, g/kWh
    ):
        self.length = length
        self.beam = beam
        self.form_factor = form_factor
        self.wind_drag_coeff = wind_drag_coeff
        self.propulsive_efficiency = propulsive_efficiency
        self.sfoc = sfoc
        # Mumford's wetted surface estimate.
        self.wetted_surface = 1.025 * length * (block_coeff * beam + 1.7 * draft)
        self.frontal_area = beam * windage_height

    def calm_water_resistance(self, speed_knots):
        if speed_knots <= 0:
            return 0.0
        v = speed_knots * KNOT
        cf = 0.075 / (math.log10(v * self.length / NU_WATER) - 2) ** 2
        return (
            0.5 * RHO_WATER * self.wetted_surface * (1 + self.form_factor) * cf * v**2
        )

    def wave_resistance(self, wave_height):
        b, length = self.beam, self.length
        return RHO_WATER * GRAVITY * wave_height**2 * b * math.sqrt(b / length) / 16

    def wind_resistance(self, speed_knots, heading_deg, u_wind, v_wind):
        heading = math.radians(heading_deg)
        tailwind = u_wind * math.sin(heading) + v_wind * math.cos(heading)
        apparent = speed_knots * KNOT - tailwind
        if apparent <= 0:
            return 0.0
        return 0.5 * RHO_AIR * self.wind_drag_coeff * self.frontal_area * apparent**2

    def fuel_rate(self, speed_knots, heading_deg, conditions):
        """Fuel burned per hour, in tonnes."""
        resistance = (
            self.calm_water_resistance(speed_knots)
            + self.wave_resistance(conditions.wave_height)
            + self.wind_resistance(
                speed_knots, heading_deg, conditions.u_wind, conditions.v_wind
            )
        )
        power_kw = resistance * speed_knots * KNOT / 1000 / self.propulsive_efficiency
        return power_kw * self.sfoc / 1e6
