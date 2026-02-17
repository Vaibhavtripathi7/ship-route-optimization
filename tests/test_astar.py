# from ship_routing.data_pipeline.loader import WeatherLoader
from ship_routing.data_pipeline.loader import WeatherLoader

from ship_routing.engine.physics import ShipPhysics
from ship_routing.engine.astar__c import AStarPlanner

def test_full_system():

    print("Generating Ocean data")
    loader = WeatherLoader(Bounds=(-10, 30, 50, 100)) # Indian Ocean
    loader.generate_synthetic_data()
    
    print("Building Ship model using physics class")
    ship = ShipPhysics(length=300, width=40, design_speed=20)
    
    print("Initializing A star algorithm for routing")
    planner = AStarPlanner(loader.dataset, ship)
    
    start = (10, 10)
    goal = (50, 50)
    
    path = planner.plan(start, goal, speed_knots=18.0)
    
    if path:
        print(f"Completed: Route calculated with {len(path)} waypoints.")
        print(f"Start: {path[0]}, End: {path[-1]}")
    else:
        print("FAILURE: No route found.")

if __name__ == "__main__":
    test_full_system()