import openai
import time
import json
import os

class DistanceCalculator:
    """Calculates travel times between POIs using OpenAI API."""
    
    def __init__(self, api_key=None):
        """Initialize the distance calculator with API key."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        openai.api_key = self.api_key
        self.cache = {}  # Cache to store previous requests
        self.request_count = 0  # Counter for API requests
        
    def get_travel_time(self, origin, destination, mode):
        """Get travel time between two POIs."""
        # Create a cache key
        cache_key = f"{origin['ID']}-{destination['ID']}-{mode}"
        
        # Check if already in cache
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Define mode of transportation
        mode_str = ["walking", "public transport", "car"][mode]
        
        # Create the API request
        prompt = (
            f"Calculate the travel time in minutes between two locations in Paris:\n"
            f"Origin: {origin['Nom']} at coordinates {origin['latitude']}, {origin['longitude']}\n"
            f"Destination: {destination['Nom']} at coordinates {destination['latitude']}, {destination['longitude']}\n"
            f"Mode of transportation: {mode_str}\n"
            f"Only respond with a single number representing the estimated travel time in minutes."
        )
        
        try:
            #print(f"API Request: {origin['Nom']} to {destination['Nom']} by {mode_str}")
            self.request_count += 1
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides precise travel time estimates in Paris."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Extract the travel time from the response
            travel_time = self._parse_time_from_response(response.choices[0].message.content)
            #print(f"Response: {travel_time} minutes")
            
            # Cache the result
            self.cache[cache_key] = travel_time
            
            return travel_time
            
        except Exception as e:
            print(f"Error getting travel time: {e}")
            # Return a fallback value
            fallback = self._fallback_travel_time(origin, destination, mode)
            self.cache[cache_key] = fallback
            return fallback
    
    def _parse_time_from_response(self, response_text):
        """Extract the travel time from API response."""
        try:
            # Try to extract just the number from the response
            time_str = ''.join(c for c in response_text if c.isdigit() or c == '.')
            return int(float(time_str))
        except:
            # If parsing fails, use a default value
            return 20
            
    def _fallback_travel_time(self, origin, destination, mode):
        """Calculate a fallback travel time if API fails."""
        # Simple calculation based on haversine distance
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(origin['latitude'])
        lon1 = math.radians(origin['longitude'])
        lat2 = math.radians(destination['latitude'])
        lon2 = math.radians(destination['longitude'])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        distance = c * r
        
        # Estimate travel time based on mode
        speeds = [5, 15, 25]  # km/h for walking, public transport, car
        time_hours = distance / speeds[mode]
        return int(time_hours * 60)  # Convert to minutes
        
    def get_request_count(self):
        """Return the number of API requests made."""
        return self.request_count