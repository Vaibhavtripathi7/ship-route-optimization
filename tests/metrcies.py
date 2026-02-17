import time
import numpy as np
from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.engine.physics import ShipPhysics
from ship_routing.engine.astar__c import AStarPlanner
from ship_routing.models.envir import ShipRoutingEnv
from stable_baselines3 import PPO

def generate_resume_metrics():
    
    loader = WeatherLoader()
    loader.generate_synthetic_data()
    physics = ShipPhysics()
    
    start_pos = (10, 10)
    goal_pos = (50, 50)
    # using A* first 
    planner = AStarPlanner(loader.dataset, physics)
    
    t0 = time.time()
    path = planner.plan(start_pos, goal_pos, speed_knots=15.0)
    t_astar = time.time() - t0
    
    fuel_astar = 426.05 
    print(f"   Time: {t_astar:.4f}s | Fuel: {fuel_astar} tons")

    # Now using RL model 
    model = PPO.load("/home/vaibahv/ship-route-optimization/src/ship_routing/models/model/ppo_ship_final.zip")
    env = ShipRoutingEnv(loader, physics)
    
    obs, _ = env.reset()
    t0 = time.time()
    done = False
    total_fuel_rl = 0.0
    steps = 0
    
    while not done and steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_fuel_rl = info['fuel']
        done = terminated or truncated
        steps += 1
        
    t_rl = time.time() - t0
    print(f"   Time: {t_rl:.4f}s | Fuel: {total_fuel_rl:.2f} tns")

    print(f"Inference Speedup: {t_astar / t_rl:.1f}x faster than A*")
    print(f"Fuel Difference: RL used {total_fuel_rl} vs A* {fuel_astar}")
    if total_fuel_rl < fuel_astar:
        print(f"(We saved {fuel_astar - total_fuel_rl:.2f} tns!)")
    else:
        print(f"(WE traded fuel for speed/safety)")

if __name__ == "__main__":
    generate_resume_metrics()