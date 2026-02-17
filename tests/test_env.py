from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.engine.physics import ShipPhysics
from ship_routing.models.envir import ShipRoutingEnv
import numpy as np

def test_simulation():
    loader = WeatherLoader()
    loader.generate_synthetic_data()
    
    physics = ShipPhysics()
    
    env = ShipRoutingEnv(loader, physics)
    print("Environment Created Successfully.")

    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    assert len(obs) == 9

    next_obs, reward, terminated, truncated, info = env.step(1)
    
    print(f"Reward Received: {reward:.4f}")
    print(f"Fuel Used so far: {info['fuel']:.4f}")
    print(f"New Observation: {next_obs}")
    
    if reward < 0:
        print("SUCCESS: Fuel consumption penalty applied correctly.")
    else:
        print("FAILURE: Reward should be negative (fuel cost).")

if __name__ == "__main__":
    test_simulation()