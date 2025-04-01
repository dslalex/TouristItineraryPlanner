import openai
import json
import os
from data.city_graph import create_graph, save_graph

def generate_city_data(city_name, api_key=None):
    """Generate tourist attraction data for a specified city using LLM."""
    if api_key:
        openai.api_key = api_key
    
    print(f"Generating data for {city_name}...")
    
    # System message to guide the AI
    system_message = """You are a travel expert with detailed knowledge of tourist attractions worldwide.
    Generate realistic and accurate tourist attractions for the specified city.
    Each attraction should include ID, name, opening hours, type, interest score (1-10), 
    visit duration in minutes, cost in euros, and accurate latitude/longitude coordinates.
    Include a mix of tourist attractions, museums, parks, landmarks, and restaurants.
    Return only valid JSON without any additional text."""
    
    # Prompt for the attractions
    user_message = f"""Generate 30 popular tourist attractions in {city_name}.
    
    Follow this exact format for each attraction:
    {{
        "ID": <unique_number>,
        "Nom": "<name>",
        "Horaire": "<opening_hours>", (format: "HH:MM-HH:MM" or "All day")
        "Type": "<type>", (either "Touristique" or "Restaurant")
        "Interet": <interest_score>, (integer from 1-10)
        "duree": <duration_minutes>, (integer representing visit time in minutes)
        "cout": <cost>, (integer or float representing cost in euros)
        "latitude": <latitude>, (accurate decimal coordinate)
        "longitude": <longitude> (accurate decimal coordinate)
    }}
    
    Return the data as a JSON array. Be precise with coordinates and realistic with opening hours, costs, and durations.
    For the first attraction, make it the most iconic landmark in {city_name}.
    Ensure IDs are sequential starting from 1."""
    
    try:
        # Make the API call
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        # Extract and parse the JSON
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Check if the response has the expected format (a list of attractions)
        attractions = None
        if "attractions" in data:
            attractions = data["attractions"]
        elif "locations" in data:
            attractions = data["locations"]
        else:
            # Try to find any array in the response
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    attractions = value
                    break
            else:
                # If no array found, assume the entire object is an array
                if isinstance(data, list):
                    attractions = data
                else:
                    attractions = list(data.values())[0] if data else []
        
        # Sanitize the attractions data to ensure correct types
        if attractions:
            sanitized_attractions = []
            for i, attraction in enumerate(attractions):
                # Create a sanitized copy
                sanitized = dict(attraction)
                
                # Ensure ID is an integer
                if "ID" in sanitized:
                    try:
                        sanitized["ID"] = int(sanitized["ID"])
                    except (ValueError, TypeError):
                        sanitized["ID"] = i + 1
                else:
                    sanitized["ID"] = i + 1
                
                # Ensure numeric fields have the correct type
                try:
                    sanitized["Interet"] = int(sanitized.get("Interet", 5))
                except (ValueError, TypeError):
                    sanitized["Interet"] = 5
                    
                try:
                    sanitized["duree"] = int(sanitized.get("duree", 60))
                except (ValueError, TypeError):
                    sanitized["duree"] = 60
                    
                try:
                    sanitized["cout"] = float(sanitized.get("cout", 0))
                except (ValueError, TypeError):
                    sanitized["cout"] = 0.0
                    
                try:
                    sanitized["latitude"] = float(sanitized.get("latitude", 0))
                    sanitized["longitude"] = float(sanitized.get("longitude", 0))
                except (ValueError, TypeError):
                    # If coordinates are invalid, use placeholder values
                    sanitized["latitude"] = 0.0
                    sanitized["longitude"] = 0.0
                
                sanitized_attractions.append(sanitized)
                
            attractions = sanitized_attractions
        
        # Create and save the graph
        if attractions:
            city_graph = create_graph(attractions, city_name)
            save_graph(city_graph, city_name)
            return city_graph
        else:
            print(f"Error: Could not parse attractions for {city_name}")
            return None
            
    except Exception as e:
        print(f"Error generating city data: {e}")
        import traceback
        traceback.print_exc()
        return None