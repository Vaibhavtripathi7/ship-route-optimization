"""Compare routing policies on held out voyages. Run from the repository root:

    python -m ship_routing.evaluate [model.zip ...]

Writes results/voyages.csv, results/summary.md and results/storm_case.png.
"""

import csv
import dataclasses
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ship_routing import geo
from ship_routing.env import MAX_SPEED, MIN_SPEED, ShipRoutingEnv, random_voyage
from ship_routing.physics import ShipPhysics, voyage_cost
from ship_routing.policies import AStarPolicy, DirectPolicy, PPOPolicy, Voyage, sail
from ship_routing.weather import CALM, INDIAN_OCEAN, Storm, WeatherField

EVAL_SEED = 10_000  # training weather uses seeds below this
CRUISE_SPEED = 14.0
BASELINE = f"direct {CRUISE_SPEED:g} kn"
HINDSIGHT = "direct at best fixed speed (hindsight)"
FIXED_SPEEDS = range(int(MIN_SPEED), int(MAX_SPEED) + 1)
RESULTS = Path("results")
CSV_FIELDS = [f.name for f in dataclasses.fields(Voyage) if f.name != "track"]


@dataclass(frozen=True)
class Scenario:
    name: str
    suite: str
    weather_seed: int
    start: tuple
    goal: tuple
    departure_hour: float
    storms: tuple | None = None  # None draws random storms from the seed

    def field(self):
        return WeatherField.synthetic(self.weather_seed, self.storms)


def storm_on_route(name, suite, seed, start, goal, departure, rng):
    """A storm that reaches the direct route when a cruising ship does."""
    fraction = rng.uniform(0.3, 0.7)
    lat = start[0] + fraction * (goal[0] - start[0])
    lon = start[1] + fraction * (goal[1] - start[1])
    eta = fraction * geo.distance_nm(*start, *goal) / CRUISE_SPEED
    drift_lat, drift_lon = rng.uniform(-0.15, 0.15, size=2)
    storm = Storm(
        lat=lat - drift_lat * (departure + eta),
        lon=lon - drift_lon * (departure + eta),
        radius_deg=rng.uniform(1.5, 3.0),
        peak_wave_m=rng.uniform(6.0, 9.0),
        peak_wind_ms=rng.uniform(15.0, 25.0),
        drift_lat=drift_lat,
        drift_lon=drift_lon,
    )
    return Scenario(name, suite, seed, start, goal, departure, (storm,))


def scenarios(open_sea=60, crossings=40):
    showcase = storm_on_route(
        "storm case",
        "showcase",
        EVAL_SEED,
        (10.0, 60.0),
        (15.0, 70.0),
        0.0,
        np.random.default_rng(EVAL_SEED),
    )
    suite = [showcase]
    for k in range(open_sea):
        seed = EVAL_SEED + 1 + k
        rng = np.random.default_rng(seed)
        start, goal = random_voyage(rng, INDIAN_OCEAN)
        departure = float(rng.uniform(0, 120))
        suite.append(
            Scenario(f"open sea {k}", "open sea", seed, start, goal, departure)
        )
    for k in range(crossings):
        seed = EVAL_SEED + 1000 + k
        rng = np.random.default_rng(seed)
        start, goal = random_voyage(rng, INDIAN_OCEAN)
        departure = float(rng.uniform(0, 120))
        suite.append(
            storm_on_route(
                f"crossing {k}", "storm crossing", seed, start, goal, departure, rng
            )
        )
    return suite


def cheapest_calm_speed(physics):
    def cost_per_nm(speed):
        return voyage_cost(physics.fuel_rate(speed, 0.0, CALM), 1.0) / speed

    return min(FIXED_SPEEDS, key=cost_per_nm)


def run(models):
    physics = ShipPhysics()
    eco = cheapest_calm_speed(physics)
    policies = [DirectPolicy(CRUISE_SPEED), DirectPolicy(eco), AStarPolicy(eco)]
    policies += [
        PPOPolicy(m, f"PPO seed {Path(m).stem.removeprefix('ppo_seed')}")
        for m in models
    ]

    results = []
    for scenario in scenarios():
        env = ShipRoutingEnv([scenario.field()], physics)
        voyage_args = (scenario.start, scenario.goal, scenario.departure_hour)
        voyages = [sail(env, p, *voyage_args) for p in policies]
        direct = [sail(env, DirectPolicy(v), *voyage_args) for v in FIXED_SPEEDS]
        arrived = [v for v in direct if v.outcome == "arrived"]
        if arrived:
            best = min(arrived, key=lambda v: v.cost_usd)
            voyages.append(dataclasses.replace(best, policy=HINDSIGHT))
        results.append((scenario, voyages))
    return results


def mean_ci(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, float("nan")
    return mean, 1.96 * statistics.stdev(values) / len(values) ** 0.5


def summary_table(results):
    by_policy = defaultdict(dict)
    for scenario, voyages in results:
        for v in voyages:
            by_policy[v.policy][scenario.name] = v
    names = [s.name for s, _ in results]
    common = [
        n
        for n in names
        if all(
            n in runs and runs[n].outcome == "arrived" for runs in by_policy.values()
        )
    ]
    if not common:
        return "No voyage was completed by every policy."

    lines = [
        f"Means over the {len(common)} voyages that every policy completed.",
        "",
        "| policy | arrived | cost k$ | fuel t | hours | cost vs "
        f"{BASELINE} | decision ms p50 / p95 |",
        "|:-|-:|-:|-:|-:|-:|-:|",
    ]
    for policy, runs in by_policy.items():
        arrived = sum(runs[n].outcome == "arrived" for n in names if n in runs)
        shared = [runs[n] for n in common]
        delta, half = mean_ci(
            [
                100 * (runs[n].cost_usd / by_policy[BASELINE][n].cost_usd - 1)
                for n in common
            ]
        )
        ms = [v.decision_ms for v in runs.values()]
        timing = (
            ""
            if policy == HINDSIGHT
            else f"{np.percentile(ms, 50):.1f} / {np.percentile(ms, 95):.1f}"
        )
        lines.append(
            f"| {policy} | {arrived}/{len(runs)} "
            f"| {statistics.mean(v.cost_usd for v in shared) / 1000:.1f} "
            f"| {statistics.mean(v.fuel_mt for v in shared):.1f} "
            f"| {statistics.mean(v.hours for v in shared):.1f} "
            f"| {delta:+.2f}% ± {half:.2f} | {timing} |"
        )
    return "\n".join(lines)


def write_report(results):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "voyages.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "suite", *CSV_FIELDS])
        for scenario, voyages in results:
            for v in voyages:
                values = (getattr(v, name) for name in CSV_FIELDS)
                writer.writerow([scenario.name, scenario.suite, *values])

    sections = [
        "# Evaluation results",
        "",
        "Generated by `python -m ship_routing.evaluate`. "
        "The ± is a 95% confidence interval on the paired mean.",
    ]
    for suite in ("open sea", "storm crossing"):
        subset = [r for r in results if r[0].suite == suite]
        sections += [
            "",
            f"## {suite.capitalize()} ({len(subset)} voyages)",
            "",
            summary_table(subset),
        ]
    (RESULTS / "summary.md").write_text("\n".join(sections) + "\n")

    scenario, voyages = next(r for r in results if r[0].suite == "showcase")
    plot_tracks(scenario, voyages, RESULTS / "storm_case.png")


def plot_tracks(scenario, voyages, path):
    field = scenario.field()
    shown = [v for v in voyages if v.policy != HINDSIGHT]
    end_hour = scenario.departure_hour + max(v.hours for v in shown)
    first = int(scenario.departure_hour // field.step_hours)
    last = int(end_hour // field.step_hours) + 1
    swath = field.wave_height[first : last + 1].max(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    filled = ax.contourf(field.lons, field.lats, swath, levels=12, cmap="Blues")
    fig.colorbar(filled, label="highest wave height during the voyage (m)")
    for v in shown:
        lats, lons = zip(*v.track)
        ax.plot(
            lons,
            lats,
            linewidth=2,
            label=f"{v.policy}: {v.cost_usd / 1000:.1f} k$, {v.hours:.0f} h",
        )
    ax.plot(*scenario.start[::-1], "ko")
    ax.plot(*scenario.goal[::-1], "k*", markersize=12)
    lats, lons = zip(*(p for v in shown for p in v.track))
    ax.set_xlim(min(lons) - 4, max(lons) + 4)
    ax.set_ylim(min(lats) - 4, max(lats) + 4)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Storm crossing the direct route")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    models = sys.argv[1:] or sorted(
        str(p) for p in Path("artifacts").glob("ppo_seed*.zip")
    )
    write_report(run(models))
    print((RESULTS / "summary.md").read_text())
