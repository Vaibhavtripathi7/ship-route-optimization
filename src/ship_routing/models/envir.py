import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.engine.physics import ShipPhysics

class ShipRoutingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, weather_loader: WeatherLoader, physics_engine: ShipPhysics):
        super(ShipRoutingEnv, self).__init__()
        
        self.weather = weather_loader
        self.physics = physics_engine
        
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.dt = 1.0        
        self.max_steps = 200 
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        
        self.current_pos = np.array([10.0, 60.0]) 
        self.goal_pos = np.array([15.0, 70.0])
        self.current_speed = 15.0 
        self.current_heading = 45.0 
        self.steps_taken = 0
        self.total_fuel = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        lat, lon = self.current_pos
        
        try:
            cond = self.weather.get_conditions(lat, lon, "2026-01-01 12:00:00")
            u_wind = cond['u_wind']
            v_wind = cond['v_wind']
            wave_h = cond['wave_height']
        except:
            u_wind, v_wind, wave_h = 0.0, 0.0, 0.0

        dist_km = np.linalg.norm(self.goal_pos - self.current_pos) * 111.0
        
        d_lat = self.goal_pos[0] - lat
        d_lon = self.goal_pos[1] - lon
        target_angle = np.degrees(np.arctan2(d_lon, d_lat)) # Angle in degrees
        
        angle_error = target_angle - self.current_heading
        angle_error = (angle_error + 180) % 360 - 180

        obs = np.array([
            lat, lon, 
            self.current_speed, self.current_heading,
            dist_km, angle_error,
            u_wind, v_wind, wave_h
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int):
        self.steps_taken += 1
        
        if action == 1: 
            self.current_heading -= 5.0
        elif action == 2: 
            self.current_heading += 5.0
        elif action == 3: 
            self.current_speed = min(self.current_speed + 1.0, 25.0) 
        elif action == 4:
            self.current_speed = max(self.current_speed - 1.0, 5.0)  
        
        self.current_heading = self.current_heading % 360

        speed_kmh = self.current_speed * 1.852
        dist_deg = (speed_kmh * self.dt) / 111.0 
        
        rad_heading = np.radians(self.current_heading)
        d_lat = dist_deg * np.cos(rad_heading) 
        d_lon = dist_deg * np.sin(rad_heading)
        
        self.current_pos[0] += d_lat
        self.current_pos[1] += d_lon

        obs = self._get_obs()
        u_wind, v_wind, wave_h = obs[6], obs[7], obs[8]
        
        weather_snapshot = {
            "u_wind": u_wind, "v_wind": v_wind, "wave_height": wave_h
        }
        
        fuel_rate_mt_h = self.physics.calculate_fuel_consumption(self.current_speed, weather_snapshot)
        fuel_consumed = fuel_rate_mt_h * self.dt
        self.total_fuel += fuel_consumed

        dist_to_goal = obs[4] 
        
        reward = -fuel_consumed
        
        terminated = False
        if dist_to_goal < 20.0: 
            reward += 1000.0 
            terminated = True
            print(f"completed! Total Fuel consumption is: {self.total_fuel:.2f} tns")

        if not (-10 <= self.current_pos[0] <= 30) or not (50 <= self.current_pos[1] <= 100):
            reward -= 100.0 
            terminated = True

        truncated = False
        if self.steps_taken >= self.max_steps:
            truncated = True

        info = {"fuel": self.total_fuel}
        
        return obs, reward, terminated, truncated, info