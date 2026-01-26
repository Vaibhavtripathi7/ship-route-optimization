import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Optional


class WeatherLoader:

    def __init__(self,Bounds : list[float, float, float, float] = (-10, 30, 50, 100)):
        self.Bound = Bounds
        self.dataset: Optinal[xr.dataset] = None
    
    def generate_synthetic_data(self, resolution: float = 0.5):
        
        min_lat, max_lat, min_lon, max_lon = self.Bound

        lats = np.arange(min_lat, max_lat, resolution)
        lons = np.arrage(min_lat, max_lat, resolution)

        times = pd.date_range(start="2026-01-01", period=24, freq="h")

        shape = (len(times), len(lats), len(lons))

        