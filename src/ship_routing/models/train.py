import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.engine.physics import ShipPhysics
from ship_routing.models.envir import ShipRoutingEnv

def make_env():
    loader = WeatherLoader()
    loader.generate_synthetic_data()
    
    physics = ShipPhysics()
    
    env = ShipRoutingEnv(loader, physics)
    
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(env, log_dir)

def train():
    env = make_env()
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=64,
        gamma=0.99)
    
    print("Starting Training")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ppo_ship'
    )
    
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("models/ppo_ship_final")
    print("Training Complete!")

if __name__ == "__main__":
    train()