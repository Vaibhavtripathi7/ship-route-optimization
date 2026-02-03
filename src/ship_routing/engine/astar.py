import heapq
import numpy as np 
from typing import List, Tuple, Dict, Optinal 
from ship_routing.engine.physics import ShipPhysics


class Node: 
    def __init__(self, lat_index, lon_index, g_cost, parent = None):
        self.lat_index = lat_index
        self.lon_index = lon_index
        self.g_cost = g_cost
        self.h_cost = 0.0
        self.parent = parent

    @property
    def fuel_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.fuel_cost < other.fuel_cost
    
class Astar:
    def __init__(self, weather_data, physics_engine):
        self.weather = weather_data
        self.physics_engine = physics_engine
        self.lats = weather_data.coords['lat'].values
        self.lons = weather_data.coords['lon'].values
        self.max_lat_idx = len(self.lats)
        self.max_lon_idx = len(self.lons)

