from ship_routing.data_pipeline.loader import WeatherLoader
import os

def test_generation():
    # 1. Instantiate
    print("Initializing Loader...")
    loader = WeatherLoader()

    # 2. Generate
    loader.generate_synthetic_data()

    # 3. Save
    save_path = "data/processed/synthetic_weather.nc"
    loader.save_data(save_path)

    # 4. Check if file exists
    if os.path.exists(save_path):
        print(f"SUCCESS: File created at {save_path}")

        # 5. Test Query
        cond = loader.get_conditions(10.0, 70.0, "2026-01-01 12:00:00")
        print(f"Sample Condition at (10N, 70E): {cond}")
    else:
        print("FAILURE: File not found.")

if __name__ == "__main__":
    test_generation()