from ortools.sat.python import cp_model
import datetime
import sys
import os
import math
from dotenv import load_dotenv

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.paris_graph import load_graph
from src.distance_api import DistanceCalculator

class TouristItinerarySolver:
    """Solver for the Tourist Trip Design Problem using constraint programming."""
    
    def __init__(self, graph=None, start_time="09:00", end_time="19:00", 
                 mandatory_visits=None, max_visits_by_type=None, api_key=None, max_neighbors=5):
        """Initialize the solver with tour parameters.
        
        Args:
            graph: NetworkX graph containing POIs and travel times
            start_time: Starting time for the tour (HH:MM format)
            end_time: Ending time for the tour (HH:MM format)
            mandatory_visits: List of POI IDs that must be visited
            max_visits_by_type: Dict of {type: max_count} for category limits
            api_key: OpenAI API key for distance calculations
            max_neighbors: Maximum number of nearest neighbors to precompute
        """
        self.graph = graph if graph else load_graph()
        self.start_time = self._time_to_minutes(start_time)
        self.end_time = self._time_to_minutes(end_time)
        self.total_available_time = self.end_time - self.start_time
        self.mandatory_visits = mandatory_visits if mandatory_visits else []
        self.max_visits_by_type = max_visits_by_type if max_visits_by_type else {}
        self.transport_mode = 1  # 0: walk, 1: public transport, 2: car
        self.distance_calculator = DistanceCalculator(api_key)
        self.max_neighbors = max_neighbors
        self.nearest_neighbors = self._precompute_nearest_neighbors()
        print(f"Precomputed {max_neighbors} nearest neighbors for each POI")
        
    def _time_to_minutes(self, time_str):
        """Convert time string (HH:MM) to minutes since midnight."""
        if time_str == "All day":
            return 0  # Special case for POIs open all day
        if "," in time_str:  # Handle multiple time intervals (e.g., lunch/dinner restaurants)
            return 0  # For simplicity, treat as always open
            
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return 0  # Default if format is incorrect
    
    def _parse_opening_hours(self, hours_str):
        """Parse opening hours string into time intervals in minutes."""
        if hours_str == "All day":
            return [(0, 24*60)]  # Open all day
            
        intervals = []
        parts = hours_str.split(", ")
        for part in parts:
            try:
                open_time, close_time = part.split("-")
                open_minutes = self._time_to_minutes(open_time)
                close_minutes = self._time_to_minutes(close_time)
                intervals.append((open_minutes, close_minutes))
            except:
                continue
        
        return intervals if intervals else [(0, 24*60)]  # Default to open all day if parsing fails
    
    def _precompute_nearest_neighbors(self):
        """Precompute the nearest neighbors for each POI based on haversine distance."""
        nearest_neighbors = {}
        pois = list(self.graph.nodes())
        
        for i in pois:
            distances = []
            for j in pois:
                if i != j:
                    # Calculate haversine distance
                    lat1 = math.radians(self.graph.nodes[i]['latitude'])
                    lon1 = math.radians(self.graph.nodes[i]['longitude'])
                    lat2 = math.radians(self.graph.nodes[j]['latitude'])
                    lon2 = math.radians(self.graph.nodes[j]['longitude'])
                    
                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    c = 2 * math.asin(math.sqrt(a))
                    r = 6371  # Radius of Earth in kilometers
                    distance = c * r
                    
                    distances.append((j, distance))
            
            # Sort by distance and keep only the closest neighbors
            distances.sort(key=lambda x: x[1])
            nearest_neighbors[i] = [poi_id for poi_id, _ in distances[:self.max_neighbors]]
        
        return nearest_neighbors
    
    def get_travel_time(self, poi_i, poi_j):
        """Get travel time between two POIs using the API."""
        # Check if we have pre-computed travel times in the edge data
        if 'travel_times' in self.graph[poi_i][poi_j]:
            return self.graph[poi_i][poi_j]['travel_times'][self.transport_mode]
        
        # If poi_j is in the nearest neighbors of poi_i, use the API
        if poi_j in self.nearest_neighbors[poi_i]:
            # Use the API to calculate travel time
            origin = self.graph.nodes[poi_i]
            destination = self.graph.nodes[poi_j]
            
            travel_time = self.distance_calculator.get_travel_time(
                origin, destination, self.transport_mode)
            
            # Store for future use
            if 'travel_times' not in self.graph[poi_i][poi_j]:
                self.graph[poi_i][poi_j]['travel_times'] = [None, None, None]
            self.graph[poi_i][poi_j]['travel_times'][self.transport_mode] = travel_time
            
            return travel_time
        else:
            # Use the fallback calculation for non-neighbors
            origin = self.graph.nodes[poi_i]
            destination = self.graph.nodes[poi_j]
            travel_time = self.distance_calculator._fallback_travel_time(origin, destination, self.transport_mode)
            
            # Store for future use
            if 'travel_times' not in self.graph[poi_i][poi_j]:
                self.graph[poi_i][poi_j]['travel_times'] = [None, None, None]
            self.graph[poi_i][poi_j]['travel_times'][self.transport_mode] = travel_time
            
            return travel_time
    
    def solve(self, max_pois=10):
        """Solve the Tourist Trip Design Problem.
        
        Args:
            max_pois: Maximum number of POIs to include in the itinerary
        
        Returns:
            A list of tuples (poi_id, arrival_time, departure_time) representing the itinerary
        """
        model = cp_model.CpModel()
        
        # Extract POIs and their attributes from the graph
        pois = list(self.graph.nodes())
        n_pois = len(pois)
        
        # Create variables
        # visit[i] = 1 if POI i is visited, 0 otherwise
        visit = {}
        for i in pois:
            visit[i] = model.NewBoolVar(f'visit_{i}')
        
        # pos[i][p] = 1 if POI i is visited at position p, 0 otherwise
        pos = {}
        for i in pois:
            pos[i] = {}
            for p in range(max_pois):
                pos[i][p] = model.NewBoolVar(f'poi_{i}_at_pos_{p}')
        
        # arrival_time[i] = arrival time at POI i in minutes from start_time
        # departure_time[i] = departure time from POI i in minutes from start_time
        max_time = self.total_available_time
        arrival_time = {}
        departure_time = {}
        for i in pois:
            arrival_time[i] = model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals([(0, max_time)]), f'arrival_{i}')
            departure_time[i] = model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals([(0, max_time)]), f'departure_{i}')
        
        # Constraint 1: Each POI is visited at most once
        for i in pois:
            model.Add(sum(pos[i][p] for p in range(max_pois)) <= 1)
            
            # Link visit[i] with pos[i][p]
            model.Add(visit[i] == sum(pos[i][p] for p in range(max_pois)))
        
        # Constraint 2: Each position has at most one POI
        for p in range(max_pois):
            model.Add(sum(pos[i][p] for i in pois) <= 1)
        
        # Constraint 3: No gaps in positions
        for p in range(1, max_pois):
            model.Add(sum(pos[i][p] for i in pois) <= sum(pos[i][p-1] for i in pois))
        
        # Constraint 4: Time constraints
        for i in pois:
            # Visit duration
            poi_duration = self.graph.nodes[i]['duree']
            
            # If POI i is visited, enforce its duration
            model.Add(departure_time[i] - arrival_time[i] >= poi_duration).OnlyEnforceIf(visit[i])
            model.Add(departure_time[i] - arrival_time[i] == 0).OnlyEnforceIf(visit[i].Not())
            
            # Opening hours constraints
            opening_intervals = self._parse_opening_hours(self.graph.nodes[i]['Horaire'])
            
            # Create a variable for each opening interval that indicates if the visit happens during this interval
            in_interval = []
            for interval_idx, (open_min, close_min) in enumerate(opening_intervals):
                open_min_relative = max(0, open_min - self.start_time)
                close_min_relative = min(max_time, close_min - self.start_time)
                
                # Skip invalid intervals
                if open_min_relative >= close_min_relative:
                    continue
                
                interval_var = model.NewBoolVar(f'poi_{i}_in_interval_{interval_idx}')
                in_interval.append(interval_var)
                
                # If interval_var is true, then the visit must be entirely within this interval
                model.Add(arrival_time[i] >= open_min_relative).OnlyEnforceIf(interval_var)
                model.Add(departure_time[i] <= close_min_relative).OnlyEnforceIf(interval_var)
            
            # If POI is visited, it must be during one of its opening intervals
            if in_interval:  # Only add this constraint if there are valid intervals
                model.Add(sum(in_interval) == visit[i])
        
        # Constraint 5: Travel time between consecutive POIs
        for p in range(max_pois - 1):
            for i in pois:
                for j in pois:
                    if i != j:
                        # Get travel time from i to j using API
                        travel_time = self.get_travel_time(i, j)
                        
                        # If i is at position p and j is at position p+1,
                        # then arrival_time[j] >= departure_time[i] + travel_time
                        i_at_p = pos[i][p]
                        j_at_p_plus_1 = pos[j][p+1]
                        
                        model.Add(arrival_time[j] >= departure_time[i] + travel_time).OnlyEnforceIf([i_at_p, j_at_p_plus_1])
        
        # Constraint 6: Total time must not exceed available time
        model.Add(sum(departure_time[i] - arrival_time[i] for i in pois) <= self.total_available_time)
        
        # Constraint 7: Mandatory visits
        for poi_id in self.mandatory_visits:
            if poi_id in pois:
                model.Add(visit[poi_id] == 1)
        
        # Constraint 8: Category limits
        for poi_type, max_count in self.max_visits_by_type.items():
            type_pois = [i for i in pois if self.graph.nodes[i].get('Type') == poi_type]
            model.Add(sum(visit[i] for i in type_pois) <= max_count)
        
        # Objective: Maximize total interest score
        objective = sum(visit[i] * self.graph.nodes[i]['Interet'] for i in pois)
        model.Maximize(objective)
        
        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60  # Set a time limit
        status = solver.Solve(model)
        
        # Check if a solution was found
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract the solution
            itinerary = []
            
            # Find the number of POIs in the solution
            n_visited = sum(solver.Value(visit[i]) for i in pois)
            
            # Extract the POIs in order
            for p in range(max_pois):
                for i in pois:
                    if solver.Value(pos[i][p]) == 1:
                        arrival = solver.Value(arrival_time[i])
                        departure = solver.Value(departure_time[i])
                        itinerary.append((i, arrival, departure))
            
            return itinerary
        else:
            return None

    def format_itinerary(self, itinerary):
        """Format the solution as a readable itinerary.
        
        Args:
            itinerary: List of (poi_id, arrival_time, departure_time) tuples
            
        Returns:
            A string representing the formatted itinerary
        """
        if not itinerary:
            return "No feasible itinerary found."
            
        result = "Optimized Tourist Itinerary:\n"
        result += "===========================\n\n"
        
        total_interest = 0
        
        for idx, (poi_id, arrival, departure) in enumerate(itinerary):
            poi = self.graph.nodes[poi_id]
            
            # Convert minutes to HH:MM format
            arrival_time = self._minutes_to_time_str(arrival + self.start_time)
            departure_time = self._minutes_to_time_str(departure + self.start_time)
            
            result += f"{idx+1}. {poi['Nom']} ({poi['Type']})\n"
            result += f"   Arrival: {arrival_time}, Departure: {departure_time}\n"
            result += f"   Duration: {poi['duree']} minutes\n"
            result += f"   Interest: {poi['Interet']}/10\n"
            result += f"   Cost: €{poi['cout']}\n"
            
            # Add travel information if not the last POI
            if idx < len(itinerary) - 1:
                next_poi_id = itinerary[idx+1][0]
                travel_time = self.graph[poi_id][next_poi_id]['travel_times'][self.transport_mode]
                
                transport_mode_str = ["walking", "public transport", "car"][self.transport_mode]
                result += f"   Travel to next: {travel_time} minutes by {transport_mode_str}\n"
            
            result += "\n"
            total_interest += poi['Interet']
        
        # Add summary
        total_time = itinerary[-1][2] - itinerary[0][1]
        total_cost = sum(self.graph.nodes[poi_id]['cout'] for poi_id, _, _ in itinerary)
        
        result += f"Summary:\n"
        result += f"Total POIs visited: {len(itinerary)}\n"
        result += f"Total interest score: {total_interest}\n"
        result += f"Total time (including travel): {total_time} minutes\n"
        result += f"Total cost: €{total_cost}\n"
        
        return result
    
    def _minutes_to_time_str(self, minutes):
        """Convert minutes since midnight to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"


def main():
    """Main function to demonstrate the solver."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key found in .env file. Please add OPENAI_API_KEY to your .env file.")
        return
    
    # Load the graph
    graph = load_graph()
    
    # Create a solver with default parameters
    solver = TouristItinerarySolver(
        graph=graph, 
        start_time="09:00", 
        end_time="20:00",
        mandatory_visits=[1],  # Tour Eiffel is mandatory
        max_visits_by_type={"Restaurant": 2, "Touristique": 10},
        api_key=api_key,
        max_neighbors=5  # Only consider the 5 nearest POIs for API requests
    )
    
    # Solve the problem
    print("Solving the tourist itinerary planning problem...")
    itinerary = solver.solve(max_pois=8)
    
    # Print the solution
    print(solver.format_itinerary(itinerary))
    
    # Print API request statistics
    print(f"\nAPI Request Statistics:")
    print(f"Total API requests made: {solver.distance_calculator.request_count}")


if __name__ == "__main__":
    main()