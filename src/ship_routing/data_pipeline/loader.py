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
        lons = np.arange(min_lon, max_lon, resolution)

        times = pd.date_range(start="2026-01-01", periods=24, freq="h")

        shape = (len(times), len(lats), len(lons))

        u_wind = np.random.normal(0, 5, size=shape) 
        v_wind = np.random.normal(0,5, size=shape)

        lat_factor = np.sin(np.linspace(0,np.pi, len(lats)))
        lat_factor = lat_factor.reshape(1,-1,1)
        u_wind += (5*lat_factor)
        wind_magnitude = np.sqrt(u_wind**2 + v_wind**2)

        wave_height = 0.2* wind_magnitude + np.random.normal(0,0.5, size=shape)
        wave_height = np.maximum(wave_height, 0)
    
        self.dataset = xr.Dataset(
            data_vars = {
                "u_wind": (["time", "lat", "lon"], u_wind),
                "v_wind": (["time", "lat", "lon"], v_wind),
                "wave_height" : (['time', 'lat','lon'], wave_height)

            },
            coords = {
                'time':times,
                'lat': lats,
                'lon': lons
    
            },
            attrs = {
                'desc': "Synthetic ocean Data",
                'units': 'm/s for wind, m for waves'
            }
        )
    print('Dataset generated!')

    def save_data(self, filepath):
        if self.dataset is None : 
            raise ValueError('No dataset')

        self.dataset.to_netcdf(filepath)
        print('data saved!')

    def get_conditions(self, lat:float, lon:float, time:str):

        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        
        # "Nearest" interpolation finds the closest grid point to the ship
        point = self.dataset.sel(
            lat=lat, 
            lon=lon, 
            time=pd.to_datetime(time), 
            method="nearest"
        )
        return {
            "u_wind": float(point["u_wind"]),
            "v_wind": float(point["v_wind"]),
            "wave_height": float(point["wave_height"])
        }