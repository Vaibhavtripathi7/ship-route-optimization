import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.engine.physics import ShipPhysics
from ship_routing.models.envir import ShipRoutingEnv

def simulate_waypoint_route(env, waypoints, speed_knots=15.0):
    """Simulates a ship traveling linearly, calculating fuel using true Nautical Miles."""
    total_fuel = 0.0
    path = []
    
    for i in range(len(waypoints) - 1):
        start_wp = waypoints[i]
        end_wp = waypoints[i+1]
        
        avg_lat = np.radians((start_wp[0] + end_wp[0]) / 2.0)
        lat_diff_nm = (end_wp[0] - start_wp[0]) * 60.0
        lon_diff_nm = (end_wp[1] - start_wp[1]) * 60.0 * np.cos(avg_lat)
        segment_dist_nm = np.sqrt(lat_diff_nm**2 + lon_diff_nm**2)
        
        steps = max(int(segment_dist_nm / 5.0), 5) 
        step_dist_nm = segment_dist_nm / steps
        time_hours_per_step = step_dist_nm / speed_knots
        
        for step in range(steps):
            fraction = step / steps
            lat = start_wp[0] + fraction * (end_wp[0] - start_wp[0])
            lon = start_wp[1] + fraction * (end_wp[1] - start_wp[1])
            path.append((lat, lon))
            
            try:
                cond = env.weather.get_conditions(lat, lon, "2026-01-01 12:00:00")
            except:
                cond = {"u_wind":0, "v_wind":0, "wave_height":0}
            
            fuel_rate = env.physics.calculate_fuel_consumption(speed_knots, cond)
            total_fuel += fuel_rate * time_hours_per_step
            
    return path, total_fuel
def run_scenario():
    loader = WeatherLoader()
    loader.generate_synthetic_data()
    
    lons_grid, lats_grid = np.meshgrid(loader.dataset.lon, loader.dataset.lat)
    
    storm_lat, storm_lon = 12.5, 65.0
    dx = lons_grid - storm_lon
    dy = lats_grid - storm_lat
    
    angle = np.radians(35)
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    
    dist = np.sqrt((rx / 8.0)**2 + (ry / 4.0)**2)
    
    storm_waves = 6.0 * np.exp(-dist**2)
    loader.dataset['wave_height'][0, :, :] += storm_waves  

    start_pos = (10.0, 60.0)
    goal_pos = (15.0, 70.0)
    
    traditional_waypoints = [start_pos, (12.5, 65.0), goal_pos]
    
    physics = ShipPhysics()
    env = ShipRoutingEnv(loader, physics)
    
    print("Running Traditional Waypoint Baseline")
    trad_path, trad_fuel = simulate_waypoint_route(env, traditional_waypoints)
    print(f"Traditional Route Fuel: {trad_fuel:.2f} MT")

    print("\nRunning Deep RL Agent")
    model = PPO.load("./src/ship_routing/models/model/ppo_ship_final.zip")
    obs, _ = env.reset()
    rl_path = [(env.current_pos)]
    rl_fuel = 0.0
    done = False
    
    while not done and len(rl_path) < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
                
        if 'fuel' in info:
            rl_fuel = info['fuel']
            
        done = terminated or truncated
        if done:
            rl_path.append(goal_pos) 
            break
        rl_path.append(tuple(env.current_pos))
    print(f"RL Route Fuel: {rl_fuel:.2f} MT")
    print(f"Fuel Saved: {trad_fuel - rl_fuel:.2f} MT")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    lons, lats = np.meshgrid(loader.dataset.lon, loader.dataset.lat)
    waves = loader.dataset['wave_height'].isel(time=0).values
    c = ax.contourf(lons, lats, waves, cmap='Blues', alpha=0.8)
    plt.colorbar(c, label='Wave Height (meters)')
    
    trad_lats, trad_lons = zip(*trad_path)
    rl_lats, rl_lons = zip(*rl_path)
    # rl_lats = [pos[0] for pos in rl_path]
    # rl_lons = [pos[1] for pos in rl_path]
    
    ax.plot(trad_lons, trad_lats, 'r--', linewidth=2, label=f'Traditional Waypoints ({trad_fuel:.1f} MT)')
    ax.scatter([wp[1] for wp in traditional_waypoints], [wp[0] for wp in traditional_waypoints], c='red', s=100, marker='X', zorder=5)
    
    ax.plot(rl_lons, rl_lats, 'g-', linewidth=3, label=f'RL Optimized Route ({rl_fuel:.1f} MT)')
    
    ax.set_title("Deep RL vs Traditional Waypoint Navigation\n(Storm Avoidance Scenario)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig("route_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved plot to route_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_scenario()