"""Train a PPO router on random voyages. Run from the repository root:

python -m ship_routing.train [seed] [timesteps]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results

from ship_routing.env import ShipRoutingEnv
from ship_routing.weather import WeatherField

TRAIN_WEATHER_SEEDS = range(24)  # evaluation draws its weather from 10000 up
N_ENVS = 8


def train(seed=0, timesteps=1_000_000):
    fields = [WeatherField.synthetic(s) for s in TRAIN_WEATHER_SEEDS]
    log_dir = Path("runs") / f"seed{seed}"
    env = make_vec_env(
        lambda: ShipRoutingEnv(fields),
        n_envs=N_ENVS,
        seed=seed,
        monitor_dir=str(log_dir),
        monitor_kwargs={"info_keywords": ("outcome",)},
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        # Voyage cost is not discounted. With gamma below one, progress credit
        # earned early outweighs fuel burned late and the agent sails too fast.
        gamma=1.0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)
    model_path = Path("artifacts") / f"ppo_seed{seed}.zip"
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    plot_learning_curve(log_dir, Path("results") / f"learning_curve_seed{seed}.png")
    return model_path


def plot_learning_curve(log_dir, path, window=200):
    episodes = load_results(str(log_dir))
    steps = episodes["l"].cumsum()
    arrived = (episodes["outcome"] == "arrived").astype(float)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    top.plot(steps, episodes["r"].rolling(window).mean())
    top.set_ylabel("episode return")
    bottom.plot(steps, arrived.rolling(window).mean())
    bottom.set_ylabel("arrival rate")
    bottom.set_xlabel("environment steps")
    bottom.set_ylim(0, 1.05)
    fig.suptitle(f"PPO training, rolling mean over {window} episodes")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    print(train(seed, timesteps))
