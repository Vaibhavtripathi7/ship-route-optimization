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

