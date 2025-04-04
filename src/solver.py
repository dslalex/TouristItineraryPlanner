from ortools.sat.python import cp_model
import datetime
import sys
import os
import math
from dotenv import load_dotenv

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.city_graph import load_graph
from src.distance_api import DistanceCalculator
from src.city_generator import generate_city_data

class TouristItinerarySolver:
    """Solver for the Tourist Trip Design Problem using constraint programming."""
    
    def __init__(self, city="paris", graph=None, start_time="09:00", end_time="19:00", 
                 mandatory_visits=None, max_visits_by_type=None, api_key=None, max_neighbors=3,
                 mandatory_restaurant=False):
        """Initialize the solver with tour parameters."""
        self.city = city.lower()
        
        # Try to load the city graph
        if graph is None:
            self.graph = load_graph(city)
            
            # If city graph doesn't exist, generate it
            if self.graph is None and api_key:
                print(f"No data found for {city}. Generating new city data...")
                self.graph = generate_city_data(city, api_key)
                
                if self.graph is None:
                    raise ValueError(f"Could not generate data for {city}. Please try again.")
            elif self.graph is None:
                raise ValueError(f"No data found for {city} and no API key provided to generate data.")
        else:
            self.graph = graph
        
        # Ensure all graph node IDs are integers
        self._ensure_integer_nodes()
        
        # Setting tour parameters
        self.start_time = self._time_to_minutes(start_time)
        self.end_time = self._time_to_minutes(end_time)
        self.total_available_time = self.end_time - self.start_time
        
        # Setting visit constraints
        self.mandatory_visits = mandatory_visits or []
        self.mandatory_restaurant = mandatory_restaurant
        # Ensure mandatory_visits are integers
        if self.mandatory_visits:
            new_mandatory = []
            for poi_id in self.mandatory_visits:
                try:
                    new_mandatory.append(int(poi_id))
                except (ValueError, TypeError):
                    print(f"Warning: Skipping mandatory visit {poi_id} - not a valid integer")
            self.mandatory_visits = new_mandatory
            
            # Verify mandatory visits exist in the graph
            for poi_id in self.mandatory_visits:
                if poi_id not in self.graph.nodes():
                    print(f"Warning: Mandatory POI {poi_id} not found in graph")
        self.max_visits_by_type = max_visits_by_type or {}
        
        # Travel parameters
        self.api_key = api_key
        self.max_neighbors = max_neighbors
        self.transport_modes = ["walking", "public transport", "car"]
        self.walking_threshold = 1.0  # km
        self.public_transport_threshold = 5.0  # km
        
        # Create distance calculator
        self.distance_calculator = DistanceCalculator(api_key)
        
        # Precompute nearest neighbors for each POI to reduce API calls
        self.nearest_neighbors = self._precompute_nearest_neighbors()
        
        print(f"Precomputed {max_neighbors} nearest neighbors for each POI")
        print(f"Transport mode preferences: Walk -> Public Transport -> Car")
        
        # Print summary of initialization
        print(f"Initialized solver for {self.city} with {len(self.graph.nodes())} POIs")
        print(f"Mandatory visits: {self.mandatory_visits}")
        print(f"Time window: {self._minutes_to_time_str(self.start_time)} - {self._minutes_to_time_str(self.end_time)}")
    
    def _ensure_integer_nodes(self):
        """Ensure all graph node IDs are integers to prevent index errors."""
        print("Checking graph node types...")
        needs_conversion = False
        
        # Check if any node ID is not an integer
        for node in list(self.graph.nodes()):
            if not isinstance(node, int):
                needs_conversion = True
                break
        
        if not needs_conversion:
            print("All node IDs are already integers.")
            return
        
        # Create a new graph with integer node IDs
        print("Converting node IDs to integers...")
        int_graph = nx.Graph()
        
        # Copy nodes with integer IDs
        for node in self.graph.nodes():
            try:
                int_node = int(node)
                int_graph.add_node(int_node, **self.graph.nodes[node])
            except (ValueError, TypeError):
                print(f"Warning: Skipping node {node} because ID cannot be converted to integer")
        
        # Copy edges
        for u, v in self.graph.edges():
            try:
                int_u = int(u)
                int_v = int(v)
                # Copy all edge attributes
                edge_attrs = self.graph.get_edge_data(u, v)
                int_graph.add_edge(int_u, int_v, **edge_attrs)
            except (ValueError, TypeError):
                print(f"Warning: Skipping edge {u}-{v} because IDs cannot be converted to integers")
        
        # Replace the graph
        self.graph = int_graph
        print(f"Graph now has {len(self.graph.nodes())} nodes with integer IDs.")
        
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
    
    def get_travel_time(self, poi_i, poi_j, mode=None):
        """Get travel time between two POIs in minutes."""
        # Ensure POI IDs are integers
        poi_i = int(poi_i)
        poi_j = int(poi_j)
        
        # If same POI, no travel time
        if poi_i == poi_j:
            return 0
        
        # Check if poi_j is among the nearest neighbors of poi_i
        # If not, return a very large travel time to discourage this connection
        if poi_j not in self.nearest_neighbors.get(poi_i, []):
            return 9999  # Large penalty value
        
        # Handle mode selection or default
        if mode is None:
            preferred_mode, travel_time = self._select_preferred_transport_mode(poi_i, poi_j)
            return travel_time
        
        mode_index = mode
        if isinstance(mode, str):
            mode_map = {"walking": 0, "public_transport": 1, "car": 2}
            mode_index = mode_map.get(mode.lower(), 0)
        
        if not isinstance(mode_index, int) or mode_index < 0 or mode_index > 2:
            print(f"Warning: Invalid mode_index {mode_index}, defaulting to 0")
            mode_index = 0
        
        if 'travel_times' not in self.graph[poi_i][poi_j]:
            self.graph[poi_i][poi_j]['travel_times'] = [None, None, None]
        
        # For walking (mode_index 0), calculate time based on haversine distance
        if mode_index == 0 and self.graph[poi_i][poi_j]['travel_times'][mode_index] is None:
            # Calculate walking time based on distance (average walking speed ~5 km/h)
            distance_km = self._calculate_haversine_distance(
                self.graph.nodes[poi_i]['latitude'], 
                self.graph.nodes[poi_j]['longitude'],
                self.graph.nodes[poi_j]['latitude'], 
                self.graph.nodes[poi_j]['longitude']
            )
            # Apply a 1.3x penalty factor to account for non-straight paths
            # 5 km/h = 12 minutes per km, multiply by penalty factor
            walking_time = int(distance_km * 12 * 1.3)
            self.graph[poi_i][poi_j]['travel_times'][mode_index] = walking_time
        # For other transport modes, use the API
        elif self.graph[poi_i][poi_j]['travel_times'][mode_index] is None:
            origin = self.graph.nodes[poi_i]
            destination = self.graph.nodes[poi_j]
            travel_time = self.distance_calculator.get_travel_time(origin, destination, mode_index)
            self.graph[poi_i][poi_j]['travel_times'][mode_index] = travel_time
        
        return self.graph[poi_i][poi_j]['travel_times'][mode_index]

    def _select_preferred_transport_mode(self, poi_i, poi_j):
        """Select the preferred transport mode based on distance and travel time heuristics."""
        # Ensure POI IDs are integers
        poi_i = int(poi_i)
        poi_j = int(poi_j)
        
        # Initialize travel_times if it doesn't exist
        if 'travel_times' not in self.graph[poi_i][poi_j]:
            self.graph[poi_i][poi_j]['travel_times'] = [None, None, None]  # One slot for each transport mode
        
        travel_times = self.graph[poi_i][poi_j]['travel_times']
        
        # Calculate haversine distance first
        distance_km = self._calculate_haversine_distance(
            self.graph.nodes[poi_i]['latitude'], 
            self.graph.nodes[poi_i]['longitude'],
            self.graph.nodes[poi_j]['latitude'], 
            self.graph.nodes[poi_j]['longitude']
        )
        
        # For walking, always calculate time based on distance instead of using API
        if travel_times[0] is None:
            # Calculate walking time: 5 km/h = 12 minutes per km, with 1.3x penalty factor
            walking_time = int(distance_km * 12 * 1.3)
            travel_times[0] = walking_time
        
        # Fast decision for very short distances - choose walking without API calls
        if distance_km <= self.walking_threshold:
            walk_time = travel_times[0]
            if walk_time <= 25:  # 25 minutes is reasonable walking time
                # Store the selected transport mode
                self.graph[poi_i][poi_j]['selected_mode'] = 0
                mode_names = ["walking", "public transport", "car"]
                #print(f"Selected {mode_names[0]} for travel from {self.graph.nodes[poi_i]['Nom']} to {self.graph.nodes[poi_j]['Nom']}")
                return 0, walk_time
        
        # For other transport modes, calculate only if needed
        for mode_idx in range(1, 3):  # Skip walking (mode_idx 0)
            if travel_times[mode_idx] is None:
                origin = self.graph.nodes[poi_i]
                destination = self.graph.nodes[poi_j]
                travel_times[mode_idx] = self.distance_calculator.get_travel_time(origin, destination, mode_idx)
        
        # Create a copy of travel times with car penalty for decision making
        decision_travel_times = travel_times.copy()
        # Apply 3x penalty to car travel time in the decision calculation
        if decision_travel_times[2] is not None:
            decision_travel_times[2] = decision_travel_times[2] * 3
        
        # Get chosen transport mode and travel time
        chosen_mode = 1  # Default to public transport
        chosen_time = travel_times[1]
        
        # Priority 2: Public transport for medium distances
        if distance_km <= self.public_transport_threshold:
            # Compare public transport with penalized car time
            if decision_travel_times[1] <= decision_travel_times[2]:
                chosen_mode = 1
                chosen_time = travel_times[1]
            else:
                chosen_mode = 2
                chosen_time = travel_times[2]
        
        # Priority 3: Car for longer distances, but only if significantly better than public transport
        else:
            if decision_travel_times[2] < decision_travel_times[1]:
                chosen_mode = 2
                chosen_time = travel_times[2]
            else:
                chosen_mode = 1
                chosen_time = travel_times[1]
        
        # Log the chosen transport mode
        mode_names = ["walking", "public transport", "car"]
        #print(f"Selected {mode_names[chosen_mode]} for travel from {self.graph.nodes[poi_i]['Nom']} to {self.graph.nodes[poi_j]['Nom']}")
        
        # Store the selected transport mode
        self.graph[poi_i][poi_j]['selected_mode'] = chosen_mode
        
        # Return both the chosen mode and the travel time
        return chosen_mode, chosen_time

    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the haversine distance between two points in kilometers."""
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def solve(self, max_pois=10):
        """Solve the Tourist Trip Design Problem.
        
        Args:
            max_pois: Maximum number of POIs to include in the itinerary
        
        Returns:
            A list of tuples (poi_id, arrival_time, departure_time) representing the itinerary
        """
        # Ensure all POIs have integer IDs
        pois = []
        for node in self.graph.nodes():
            try:
                pois.append(int(node))
            except (ValueError, TypeError):
                print(f"Warning: Skipping POI {node} - not a valid integer ID")

        print(f"Solving model with {len(pois)} POIs...")
        
        model = cp_model.CpModel()
        
        # Extract POIs and their attributes from the graph
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
        
        # Constraint for mandatory restaurant
        if self.mandatory_restaurant:
            restaurant_pois = [i for i in pois if self.graph.nodes[i].get('Type') == "Restaurant"]
            if restaurant_pois:
                model.Add(sum(visit[i] for i in restaurant_pois) >= 1)
                
                # Enforce meal time constraints for restaurants
                lunch_start = self._time_to_minutes("11:00") - self.start_time
                lunch_end = self._time_to_minutes("13:00") - self.start_time
                dinner_start = self._time_to_minutes("19:00") - self.start_time
                dinner_end = self._time_to_minutes("21:00") - self.start_time
                
                for i in restaurant_pois:
                    # Variable indicating if restaurant i is visited during lunch
                    lunch_visit = model.NewBoolVar(f'lunch_visit_{i}')
                    
                    # Variable indicating if restaurant i is visited during dinner
                    dinner_visit = model.NewBoolVar(f'dinner_visit_{i}')
                    
                    # Link the time constraints to these variables
                    model.Add(arrival_time[i] >= lunch_start).OnlyEnforceIf(lunch_visit)
                    model.Add(arrival_time[i] <= lunch_end - self.graph.nodes[i]['duree']).OnlyEnforceIf(lunch_visit)
                    
                    model.Add(arrival_time[i] >= dinner_start).OnlyEnforceIf(dinner_visit)
                    model.Add(arrival_time[i] <= dinner_end - self.graph.nodes[i]['duree']).OnlyEnforceIf(dinner_visit)
                    
                    # Restaurant visits must be during lunch or dinner
                    model.Add(lunch_visit + dinner_visit == visit[i])
        
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
        """Format the solution as a readable itinerary."""
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
                
                # Get the selected transport mode and travel time
                if 'selected_mode' in self.graph[poi_id][next_poi_id]:
                    selected_mode = self.graph[poi_id][next_poi_id]['selected_mode']
                else:
                    selected_mode = 1  # Default to public transport
                    
                travel_time = self.graph[poi_id][next_poi_id]['travel_times'][selected_mode]
                
                transport_mode_str = ["walking", "public transport", "car"][selected_mode]
                result += f"   Travel to next: {travel_time} minutes by {transport_mode_str}\n"
            
            result += "\n"
            total_interest += poi['Interet']
        
        # Add summary
        total_time = itinerary[-1][2] - itinerary[0][1]
        total_cost = sum(self.graph.nodes[poi_id]['cout'] for poi_id, _, _ in itinerary)
        
        # Calculate transport mode distribution
        transport_counts = {"walking": 0, "public transport": 0, "car": 0}
        for idx in range(len(itinerary) - 1):
            current_poi_id = itinerary[idx][0]
            next_poi_id = itinerary[idx+1][0]
            
            if 'selected_mode' in self.graph[current_poi_id][next_poi_id]:
                selected_mode = self.graph[current_poi_id][next_poi_id]['selected_mode']
                mode_name = ["walking", "public transport", "car"][selected_mode]
                transport_counts[mode_name] += 1
        
        result += f"Summary:\n"
        result += f"Total POIs visited: {len(itinerary)}\n"
        result += f"Total interest score: {total_interest}\n"
        result += f"Total time (including travel): {total_time} minutes\n"
        result += f"Total cost: €{total_cost}\n"
        result += f"Transport modes used: {transport_counts}\n"
        
        return result
    
    def _minutes_to_time_str(self, minutes):
        """Convert minutes since midnight to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"