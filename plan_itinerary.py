#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv
from src.solver import TouristItinerarySolver
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

def plan_itinerary(city, start_time="08:00", end_time="22:00", max_pois=6, restaurant_count=1, api_key=None):
    """Plan a tourist itinerary and return the results"""
    try:
        solver = TouristItinerarySolver(
            city=city,
            start_time=start_time, 
            end_time=end_time,
            mandatory_visits=[1],
            api_key=api_key,
            max_neighbors=3,
            mandatory_restaurant=True if restaurant_count > 0 else False,
            restaurant_count=restaurant_count
        )
        
        # Solve the problem
        itinerary = solver.solve(max_pois=max_pois)
        
        # Format the itinerary
        formatted_itinerary = solver.format_itinerary(itinerary)
        
        # Return results
        return {
            "success": True,
            "itinerary": formatted_itinerary,
            "raw_itinerary": itinerary,
            "stats": {
                "api_requests": solver.distance_calculator.request_count
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/plan', methods=['POST'])
def api_plan():
    """API endpoint for planning an itinerary"""
    data = request.json
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({
            "success": False,
            "error": "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
        })
    
    # Get parameters from request
    city = data.get('city')
    start_time = data.get('start_time', '08:00')
    end_time = data.get('end_time', '22:00')
    max_pois = int(data.get('max_pois', 6))
    restaurant_count = int(data.get('restaurant_count', 1))
    
    # Plan the itinerary
    result = plan_itinerary(city, start_time, end_time, max_pois, restaurant_count, api_key)
    
    return jsonify(result)

@app.route('/api/fun-facts', methods=['POST'])
def api_fun_facts():
    """API endpoint for getting fun facts about a city"""
    data = request.json
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({
            "success": False,
            "error": "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
        })
    
    # Get city name from request
    city = data.get('city')
    if not city:
        return jsonify({
            "success": False,
            "error": "No city specified."
        })
    
    # Generate fun facts
    from src.city_generator import generate_city_fun_facts
    fun_facts = generate_city_fun_facts(city, api_key, count=8)
    
    return jsonify({
        "success": True,
        "fun_facts": fun_facts
    })

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Check if running as web server
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Start web server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        print(f"Starting web server on port {port}...")
        app.run(debug=True, host='0.0.0.0', port=port)
        return
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plan_itinerary.py <city_name> [start_time] [end_time] [max_pois] [restaurant_count]")
        print("       python plan_itinerary.py --server [port]")
        print("Example: python plan_itinerary.py 'Paris' '09:00' '18:00' 8 2")
        sys.exit(1)
    
    # Get city name
    city = sys.argv[1]
    
    # Get optional parameters
    start_time = sys.argv[2] if len(sys.argv) > 2 else "08:00"
    end_time = sys.argv[3] if len(sys.argv) > 3 else "22:00"
    max_pois = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    restaurant_count = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    
    print(f"Planning itinerary for {city}...")
    print(f"Time window: {start_time} to {end_time}")
    print(f"Maximum POIs: {max_pois}")
    print(f"Restaurant count: {restaurant_count}")
    
    # Plan the itinerary
    result = plan_itinerary(city, start_time, end_time, max_pois, restaurant_count, api_key)
    
    if result["success"]:
        print(result["itinerary"])
        print(f"\nAPI Request Statistics:")
        print(f"Total API requests made: {result['stats']['api_requests']}")
    else:
        print(f"Error planning itinerary: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()