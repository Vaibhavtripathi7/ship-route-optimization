import numpy as np
from typing import Dict, Union

class ShipPhysics: 


    def __init__(self, length = 200.0, width = 32.0, draft = 12.0, block_coeff = 0.8, design_speed = 15.0):
        self.length = length
        self.width = width
        self.draft = draft
        self.block_coeff = block_coeff
        self.design_speed = design_speed

        self.rho_water = 1025.0
        self.rho_air = 1.225

        self.surface_area = 1.025 * self.length * (self.block_coeff * self.width + 1.7 * self.draft)


    def knots_to_ms(self, knots):
        return knots*0.514444

    
    def get_calm_water_resistance(self, speed_knots): 
        speed_knots= self.knots_to_ms(speed_knots)
        coeff_of_resist = 0.0025 
        resistance = 0.5 * self.rho_water * self.surface_area * coeff_of_resist * (speed_knots**2)
        return resistance

    def get_wind_resistance(self, speed_knots, wind_speed):
        area_front = self.width * 10
        drag_coeff = 0.8
        speed_knots = self.knots_to_ms(speed_knots)
        wind_speed = self.knots_to_ms(wind_speed)
        vel_relative = speed_knots + wind_speed
        wind_resistance = 0.5 * self.rho_air * drag_coeff * area_front * (vel_relative**2) 

        return wind_resistance
    
    def wave_resistance(self, wave_height): 
        resistance = 1500 * (wave_height**2) * self.width
        return resistance

    def calculate_fuel_consumption(self, speed_knots, weather_data: dict[str, float]):

        u_wind = weather_data["u_wind"]
        v_wind = weather_data["v_wind"]
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)

        wave_height = weather_data["wave_height"] 
        total_resistance = self.get_calm_water_resistance(speed_knots) + self.get_wind_resistance(speed_knots, wind_speed) + self.wave_resistance(wave_height)
        speed_ms = self.knots_to_ms(speed_knots)

        effective_power = total_resistance * speed_ms
        engine_power = (effective_power/1000) / 0.7
        fuel_burn = engine_power * (180/1000000)

        return fuel_burn
    


