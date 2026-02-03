import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from ship_routing.engine.physics import ShipPhysics

class Node:
    """
    Represents a specific grid point on the map.
    """
    def __init__(self, lat_idx: int, lon_idx: int, g_cost: float, parent=None):
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.g_cost = g_cost  # Actual Fuel burned from Start -> Here
        self.h_cost = 0.0     # Heuristic: Estimated Fuel Here -> Goal
        self.parent = parent  # The node we came from (to retrace steps later)
        
    @property
    def f_cost(self):
        """Total Cost (F) = Actual Cost (G) + Estimated Cost (H)"""
        return self.g_cost + self.h_cost

    # This 'less than' method is required for the Priority Queue to sort nodes
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarPlanner:
    def __init__(self, weather_data, physics_engine: ShipPhysics):
        self.weather = weather_data
        self.physics = physics_engine
        
        # We grab the raw coordinate arrays for fast access
        self.lats = weather_data.coords['lat'].values
        self.lons = weather_data.coords['lon'].values
        
        # Grid boundaries
        self.max_lat_idx = len(self.lats)
        self.max_lon_idx = len(self.lons)

    def heuristic(self, start_idx: Tuple[int, int], goal_idx: Tuple[int, int]) -> float:
        """
        Estimates the cost to the goal.
        Engineering Decision: We use Euclidean distance weighted by a 'minimum fuel factor'.
        This ensures the heuristic is 'admissible' (never overestimates).
        """
        # Distance squared
        d_lat = start_idx[0] - goal_idx[0]
        d_lon = start_idx[1] - goal_idx[1]
        dist = np.sqrt(d_lat**2 + d_lon**2) * 111.0 # approx 111km per degree index (simplified)
        
        # Assume a very efficient ship burns 0.01 tons per km.
        return dist * 0.01

    def get_neighbors(self, node: Node) -> List[Tuple[int, int]]:
        """
        Returns a list of valid (lat_idx, lon_idx) tuples for 8 directions.
        """
        neighbors = []
        # All 8 directions: (d_lat, d_lon)
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0), # Cardinals (Up, Down, Left, Right)
            (1, 1), (1, -1), (-1, 1), (-1, -1) # Diagonals
        ]
        
        for d_lat, d_lon in directions:
            new_lat = node.lat_idx + d_lat
            new_lon = node.lon_idx + d_lon
            
            # Boundary Check: Is this point actually on the map?
            if 0 <= new_lat < self.max_lat_idx and 0 <= new_lon < self.max_lon_idx:
                neighbors.append((new_lat, new_lon))
                
        return neighbors

    def plan(self, start_idx: Tuple[int, int], goal_idx: Tuple[int, int], speed_knots: float = 15.0):
        """
        The Main Algorithm.
        """
        # 1. Initialize Priority Queue (Open List)
        open_list = []
        start_node = Node(start_idx[0], start_idx[1], 0.0)
        start_node.h_cost = self.heuristic(start_idx, goal_idx)
        
        heapq.heappush(open_list, start_node)
        
        # 2. Visited Set (Closed List)
        # We store the g_cost. If we find a cheaper way to a visited node, we update it.
        visited = {} 
        
        final_node = None
        
        # Optimization: Pre-calculate speed in km/h
        speed_kmh = speed_knots * 1.852

        print(f"Starting A* Search from {start_idx} to {goal_idx}...")

        while open_list:
            # Pop the node with the lowest F-Cost
            current = heapq.heappop(open_list)
            
            # Check if we reached the goal
            if (current.lat_idx, current.lon_idx) == goal_idx:
                final_node = current
                break

            # If we've been here before with a cheaper cost, skip processing
            current_pos = (current.lat_idx, current.lon_idx)
            if current_pos in visited and visited[current_pos] <= current.g_cost:
                continue
            
            # Mark as visited
            visited[current_pos] = current.g_cost

            # Explore Neighbors
            for n_lat, n_lon in self.get_neighbors(current):
                
                # --- PHYSICS INTEGRATION START ---
                # 1. Get Weather at Neighbor
                # We use .isel() to select by index. Time=0 is hardcoded for now.
                # .item() converts numpy scalar to python float
                try:
                    u_wind = self.weather['u_wind'].isel(time=0, lat=n_lat, lon=n_lon).item()
                    v_wind = self.weather['v_wind'].isel(time=0, lat=n_lat, lon=n_lon).item()
                    wave_h = self.weather['wave_height'].isel(time=0, lat=n_lat, lon=n_lon).item()
                except Exception:
                    # Skip if data is bad
                    continue

                weather_snapshot = {
                    "u_wind": u_wind, 
                    "v_wind": v_wind, 
                    "wave_height": wave_h
                }

                # 2. Calculate Fuel Rate (Tons/Hour)
                fuel_rate_mt_h = self.physics.calculate_fuel_consumption(speed_knots, weather_snapshot)
                
                # 3. Calculate Travel Time
                # Simplified distance: 1 grid unit approx 55km (0.5 deg * 111km)
                # Diagonal moves are sqrt(2) longer
                is_diagonal = (current.lat_idx != n_lat) and (current.lon_idx != n_lon)
                dist_km = 55.0 * (1.414 if is_diagonal else 1.0)
                
                time_hours = dist_km / speed_kmh
                
                # 4. Total Cost for this Step (Fuel Burned)
                step_fuel_cost = fuel_rate_mt_h * time_hours
                # --- PHYSICS INTEGRATION END ---

                # Create new neighbor node
                new_g_cost = current.g_cost + step_fuel_cost
                new_node = Node(n_lat, n_lon, new_g_cost, parent=current)
                new_node.h_cost = self.heuristic((n_lat, n_lon), goal_idx)
                
                # Only add to queue if we haven't found a better path there already
                if (n_lat, n_lon) not in visited or new_g_cost < visited[(n_lat, n_lon)]:
                     heapq.heappush(open_list, new_node)

        # Path Reconstruction
        if final_node:
            print(f"Goal Reached! Total Fuel: {final_node.g_cost:.2f} tons")
            path = []
            curr = final_node
            while curr:
                path.append((curr.lat_idx, curr.lon_idx))
                curr = curr.parent
            return path[::-1] # Reverse list to get Start -> Goal
        else:
            print("No path found.")
            return []