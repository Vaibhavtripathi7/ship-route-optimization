import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from ship_routing.engine.physics import ShipPhysics

class Node:
    def __init__(self, lat_idx: int, lon_idx: int, g_cost: float, parent=None):
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.g_cost = g_cost  
        self.h_cost = 0.0     
        self.parent = parent  

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarPlanner:
    def __init__(self, weather_data, physics_engine: ShipPhysics):
        self.weather = weather_data
        self.physics = physics_engine
        
        self.lats = weather_data.coords['lat'].values
        self.lons = weather_data.coords['lon'].values
        
        self.max_lat_idx = len(self.lats)
        self.max_lon_idx = len(self.lons)

    def heuristic(self, start_idx: Tuple[int, int], goal_idx: Tuple[int, int]) -> float:

        d_lat = start_idx[0] - goal_idx[0]
        d_lon = start_idx[1] - goal_idx[1]
        dist = np.sqrt(d_lat**2 + d_lon**2) * 111.0 
        
        return dist * 0.01

    def get_neighbors(self, node: Node) -> List[Tuple[int, int]]:

        neighbors = []
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0), 
            (1, 1), (1, -1), (-1, 1), (-1, -1) 
        ]
        
        for d_lat, d_lon in directions:
            new_lat = node.lat_idx + d_lat
            new_lon = node.lon_idx + d_lon
            
            if 0 <= new_lat < self.max_lat_idx and 0 <= new_lon < self.max_lon_idx:
                neighbors.append((new_lat, new_lon))
                
        return neighbors

    def smoothingcurve(self, start, end):
        x0, y0 = start.lat_idx, start.lon_idx
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1- y0)
        x, y = x0, y0

        sx = 1 if x0 < x1 else -1 
        sy = 1 if y0 < y1 else -1 
        err = dx - dy

        while True:
            try:
                if not (0 <= x < self.max_lat_idx) and (0 <= y < self.max_lon_idx):
                    return False
            
            except: 
                return False

            if (x,y) == (x1, y1):
                return True

            e2 = 2 * err
            if e2 > -dy:
                err -= dy 
            elif e2 < dx:
                err += dx 
                y += sy

        return True

    
    def plan(self, start_idx: Tuple[int, int], goal_idx: Tuple[int, int], speed_knots: float = 15.0):

        open_list = []
        start_node = Node(start_idx[0], start_idx[1], 0.0)
        start_node.h_cost = self.heuristic(start_idx, goal_idx)
        
        heapq.heappush(open_list, start_node)
        
        visited = {} 
        
        final_node = None
        
        speed_kmh = speed_knots * 1.852

        print(f"Implementing A* algorithm Search from {start_idx} to {goal_idx}")

        while open_list:
            current = heapq.heappop(open_list)
            
            if (current.lat_idx, current.lon_idx) == goal_idx:
                final_node = current
                break

            current_pos = (current.lat_idx, current.lon_idx)
            if current_pos in visited and visited[current_pos] <= current.g_cost:
                continue
            
            visited[current_pos] = current.g_cost

            for n_lat, n_lon in self.get_neighbors(current):
                
                try:
                    u_wind = self.weather['u_wind'].isel(time=0, lat=n_lat, lon=n_lon).item()
                    v_wind = self.weather['v_wind'].isel(time=0, lat=n_lat, lon=n_lon).item()
                    wave_h = self.weather['wave_height'].isel(time=0, lat=n_lat, lon=n_lon).item()
                except Exception:
                    continue

                weather_snapshot = {
                    "u_wind": u_wind, 
                    "v_wind": v_wind, 
                    "wave_height": wave_h
                }

                fuel_rate_mt_h = self.physics.calculate_fuel_consumption(speed_knots, weather_snapshot)
                
                is_diagonal = (current.lat_idx != n_lat) and (current.lon_idx != n_lon)
                dist_km = 55.0 * (1.414 if is_diagonal else 1.0)
                
                time_hours = dist_km / speed_kmh
                
                step_fuel_cost = fuel_rate_mt_h * time_hours

                new_g_cost = current.g_cost + step_fuel_cost
                new_node = Node(n_lat, n_lon, new_g_cost, parent=current)
                new_node.h_cost = self.heuristic((n_lat, n_lon), goal_idx)
                
                if (n_lat, n_lon) not in visited or new_g_cost < visited[(n_lat, n_lon)]:
                     heapq.heappush(open_list, new_node)

        if final_node:
            print(f"Completed,Total Fuel consumption is: {final_node.g_cost:.2f} tns")
            path = []
            curr = final_node
            while curr:
                path.append((curr.lat_idx, curr.lon_idx))
                curr = curr.parent
            return path[::-1]
        else:
            print("No path found.")
            return []