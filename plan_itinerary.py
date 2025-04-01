#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv
from src.solver import TouristItinerarySolver

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plan_itinerary.py <city_name> [start_time] [end_time] [max_pois]")
        print("Example: python plan_itinerary.py 'Paris' '09:00' '18:00' 8")
        sys.exit(1)
    
    # Get city name
    city = sys.argv[1]
    
    # Get optional parameters
    start_time = sys.argv[2] if len(sys.argv) > 2 else "09:00"
    end_time = sys.argv[3] if len(sys.argv) > 3 else "20:00"
    max_pois = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    
    print(f"Planning itinerary for {city}...")
    print(f"Time window: {start_time} to {end_time}")
    print(f"Maximum POIs: {max_pois}")
    
    try:
        # Create and run the solver
        solver = TouristItinerarySolver(
            city=city,
            start_time=start_time, 
            end_time=end_time,
            mandatory_visits=[1],  # First attraction is mandatory
            max_visits_by_type={"Restaurant": 2, "Touristique": 10},
            api_key=api_key,
            max_neighbors=5
        )
        
        # Solve the problem
        print("Solving the tourist itinerary planning problem...")
        itinerary = solver.solve(max_pois=max_pois)
        
        # Print the solution
        print(solver.format_itinerary(itinerary))
        
        # Print API request statistics
        print(f"\nAPI Request Statistics:")
        print(f"Total API requests made: {solver.distance_calculator.request_count}")
        
    except Exception as e:
        print(f"Error planning itinerary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()