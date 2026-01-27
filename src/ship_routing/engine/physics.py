import numpy as np
from typing import Dict, Union

class ShipPhysics: 


    def __init__(self, length, width, draft, block_coeff, design_speed):
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
        speed_knots= knots_to_ms(v)
        coeff_of_resist = 0.0025 # for large ships !
        resistance = 0.5 * self.rho_water * self.surface_water * coeff_of_resist * (speed_knots**2)
        return resistance

    def get_wind_resistance(self, speed_knots, wind_speed):
        area_front = self.width * 10
        drag_coeff = 0.8
        speed_knots = knots_to_ms(speed_knots)
        wind_speed = knots_to_ms(wind_speed)
        vel_relative = speed_knots + wind_speed
        wind_resistance = 0.5 * self.rho_air * drag_coeff * (vel_relative**2) 

        return wind_resistance
    
    def calculate_fuel_consumption(self, speed_knots, weather_data: dict[str, float]):

        

