# from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.data_pipeline.loader import WeatherLoader

from ship_routing.engine.physics import ShipPhysics
from ship_routing.engine.astar__c import AStarPlanner

def test_full_system():
    # 1. Create the Ocean
    print("1. Generating Ocean...")
    loader = WeatherLoader(Bounds=(-10, 30, 50, 100)) # Indian Ocean
    loader.generate_synthetic_data()
    
    # 2. Create the Ship
    print("2. Building Ship...")
    ship = ShipPhysics(length=300, width=40, design_speed=20)
    
    # 3. Initialize Captain (A*)
    print("3. Initializing AI Planner...")
    planner = AStarPlanner(loader.dataset, ship)
    
    # 4. Plan a Route
    # Indices: Start (10, 10) -> Goal (50, 50)
    start = (10, 10)
    goal = (50, 50)
    
    path = planner.plan(start, goal, speed_knots=18.0)
    
    if path:
        print(f"SUCCESS: Route calculated with {len(path)} waypoints.")
        print(f"Start: {path[0]}, End: {path[-1]}")
    else:
        print("FAILURE: No route found.")

if __name__ == "__main__":
    test_full_system()