from ortools.sat.python import cp_model
import datetime
import sys
import os
import math
import networkx as nx
from dotenv import load_dotenv

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.city_graph import load_graph
from src.distance_api import DistanceCalculator
from src.city_generator import generate_city_data

class TouristItinerarySolver:
    """Solver for the Tourist Trip Design Problem using constraint programming."""
    
    def __init__(self, city="paris", graph=None, start_time="09:00", end_time="19:00", 
                 mandatory_visits=None, api_key=None, max_neighbors=3,
                 mandatory_restaurant=True, restaurant_count=1, max_pois=6, use_api_for_distance=True):
        """Initialize the solver with tour parameters."""
        self.city = city.lower()
        
        # Add these missing attributes
        self.max_pois = max_pois
        self.visit_duration = {}
        self.original_visit_duration = {}
        self.start_time_minutes = self._time_to_minutes(start_time)
        
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
        self.restaurant_count = restaurant_count
        
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
        
        # Travel parameters
        self.api_key = api_key
        self.max_neighbors = max_neighbors
        self.transport_modes = ["walking", "public transport", "car"]
        self.walking_threshold = 1.0  # km
        self.public_transport_threshold = 5.0  # km
        
        # Add the parameter to the constructor
        self.use_api_for_distance = use_api_for_distance
        
        # Initialize distance calculator with the parameter
        self.distance_calculator = DistanceCalculator(api_key=api_key, use_api=use_api_for_distance)
        
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
            return 0  # Open all day
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
        # Ensure POI IDs are integers
        poi_i = int(poi_i)
        poi_j = int(poi_j)
        
        # If same POI, no travel time
        if poi_i == poi_j:
            return 0
        
        # Special handling for mandatory POIs - always allow connections to them
        if poi_j in self.mandatory_visits or poi_i in self.mandatory_visits:
            # Pass through to mode selection without nearest neighbor check
            pass
        # Regular nearest neighbor check for non-mandatory POIs
        elif poi_j not in self.nearest_neighbors.get(poi_i, []):
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
    
    def solve(self, max_pois=None):
        """Solve the Tourist Trip Design Problem using an optimized algorithm from research.
        
        This implementation is based on the algorithm presented in the research paper
        which uses a two-phase approach: first selecting POIs based on interest scores
        and then optimizing their sequence with time window constraints.
        """
        # Use instance max_pois if none provided
        if max_pois is None:
            max_pois = self.max_pois
            
        # Ensure all POIs have integer IDs
        pois = []
        for node in self.graph.nodes():
            try:
                pois.append(int(node))
            except (ValueError, TypeError):
                print(f"Warning: Skipping POI {node} - not a valid integer ID")

        print(f"Solving model with {len(pois)} POIs...")
        
        # Extract POI types
        restaurant_pois = [i for i in pois if self.graph.nodes[i].get('Type') == "Restaurant"]
        tourist_pois = [i for i in pois if self.graph.nodes[i].get('Type') != "Restaurant"]
        
        print(f"Found {len(restaurant_pois)} restaurants and {len(tourist_pois)} tourist attractions")

        # Initialize the model
        model = cp_model.CpModel()
        
        # ==== Phase 1: Define core decision variables ====
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
        
        # Time variables: define domain from 0 to total available time
        max_time = self.total_available_time
        arrival_time = {}
        departure_time = {}
        for i in pois:
            arrival_time[i] = model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals([(0, max_time)]), f'arrival_{i}')
            departure_time[i] = model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals([(0, max_time)]), f'departure_{i}')
        
        # ==== Phase 2: Core structural constraints ====
        # 1. Each POI is visited at most once
        for i in pois:
            model.Add(sum(pos[i][p] for p in range(max_pois)) <= 1)
            # Link visit[i] with pos[i][p]
            model.Add(visit[i] == sum(pos[i][p] for p in range(max_pois)))
        
        # 2. Each position has at most one POI
        for p in range(max_pois):
            model.Add(sum(pos[i][p] for i in pois) <= 1)
        
        # 3. No gaps in positions
        for p in range(1, max_pois):
            model.Add(sum(pos[i][p] for i in pois) <= sum(pos[i][p-1] for i in pois))
        
        # ==== Phase 3: Visit duration and time window constraints ====
        # POI visit duration and opening hours
        for i in pois:
            poi_duration = self.graph.nodes[i]['duree']
            
            # Visit duration enforcement
            model.Add(departure_time[i] - arrival_time[i] >= poi_duration).OnlyEnforceIf(visit[i])
            model.Add(departure_time[i] - arrival_time[i] == 0).OnlyEnforceIf(visit[i].Not())
            
            # Opening hours constraints
            opening_intervals = self._parse_opening_hours(self.graph.nodes[i]['Horaire'])
            in_interval = []
            
            # For each opening interval, create a variable indicating if the visit happens during this interval
            for interval_idx, (open_min, close_min) in enumerate(opening_intervals):
                open_min_relative = max(0, open_min - self.start_time)
                close_min_relative = min(max_time, close_min - self.start_time)
                
                if open_min_relative >= close_min_relative:
                    continue
                
                interval_var = model.NewBoolVar(f'poi_{i}_in_interval_{interval_idx}')
                in_interval.append(interval_var)
                
                # If interval_var is true, then the visit must be entirely within this interval
                model.Add(arrival_time[i] >= open_min_relative).OnlyEnforceIf(interval_var)
                model.Add(departure_time[i] <= close_min_relative).OnlyEnforceIf(interval_var)
            
            # If POI is visited, it must be during one of its opening intervals
            if in_interval:
                model.Add(sum(in_interval) == visit[i])
        
        # ==== Phase 4: Travel time constraints ====
        # Travel time between consecutive POIs
        for p in range(max_pois - 1):
            # For each pair of POIs that could be visited consecutively
            for i in pois:
                for j in pois:
                    if i != j:
                        # Pre-compute travel time using the preferred transport mode
                        travel_time = self.get_travel_time(i, j)
                        
                        # Create a variable indicating if j follows i in the itinerary
                        follows = model.NewBoolVar(f'poi_{j}_follows_{i}_at_pos_{p}')
                        
                        # Link the follows variable with position variables
                        model.AddBoolAnd([pos[i][p], pos[j][p+1]]).OnlyEnforceIf(follows)
                        model.AddBoolOr([pos[i][p].Not(), pos[j][p+1].Not()]).OnlyEnforceIf(follows.Not())
                        
                        # If j follows i, enforce the travel time constraint
                        model.Add(arrival_time[j] >= departure_time[i] + travel_time).OnlyEnforceIf(follows)
        
        # ==== Phase 5: Restaurant scheduling constraints ====
        # Define meal time periods
        lunch_start = self._time_to_minutes("12:00") - self.start_time
        lunch_end = self._time_to_minutes("13:00") - self.start_time
        dinner_start = self._time_to_minutes("19:00") - self.start_time
        dinner_end = self._time_to_minutes("20:00") - self.start_time
        
        if restaurant_pois:
            # Variables for lunch and dinner visits
            lunch_visit = model.NewBoolVar('lunch_restaurant_visit')
            dinner_visit = model.NewBoolVar('dinner_restaurant_visit')
            
            # Restaurant visit constraints
            lunch_restaurants = {}
            dinner_restaurants = {}
            
            for i in restaurant_pois:
                # Define variables for each restaurant
                lunch_restaurants[i] = model.NewBoolVar(f'lunch_rest_{i}')
                dinner_restaurants[i] = model.NewBoolVar(f'dinner_rest_{i}')
                
                # Link restaurant to lunch time
                is_in_lunch_time = model.NewBoolVar(f'in_lunch_time_{i}')
                model.Add(arrival_time[i] >= lunch_start).OnlyEnforceIf(is_in_lunch_time)
                model.Add(arrival_time[i] <= lunch_end).OnlyEnforceIf(is_in_lunch_time)
                
                model.AddBoolAnd([visit[i], is_in_lunch_time]).OnlyEnforceIf(lunch_restaurants[i])
                model.AddBoolOr([visit[i].Not(), is_in_lunch_time.Not()]).OnlyEnforceIf(lunch_restaurants[i].Not())
                
                # Link restaurant to dinner time
                is_in_dinner_time = model.NewBoolVar(f'in_dinner_time_{i}')
                model.Add(arrival_time[i] >= dinner_start).OnlyEnforceIf(is_in_dinner_time)
                model.Add(arrival_time[i] <= dinner_end).OnlyEnforceIf(is_in_dinner_time)
                
                model.AddBoolAnd([visit[i], is_in_dinner_time]).OnlyEnforceIf(dinner_restaurants[i])
                model.AddBoolOr([visit[i].Not(), is_in_dinner_time.Not()]).OnlyEnforceIf(dinner_restaurants[i].Not())
            
            # Link lunch_visit and dinner_visit to existence of lunch/dinner restaurants
            model.Add(sum(lunch_restaurants[i] for i in restaurant_pois) >= 1).OnlyEnforceIf(lunch_visit)
            model.Add(sum(lunch_restaurants[i] for i in restaurant_pois) == 0).OnlyEnforceIf(lunch_visit.Not())
            
            model.Add(sum(dinner_restaurants[i] for i in restaurant_pois) >= 1).OnlyEnforceIf(dinner_visit)
            model.Add(sum(dinner_restaurants[i] for i in restaurant_pois) == 0).OnlyEnforceIf(dinner_visit.Not())
            
            # Apply restaurant count specific constraints
            if self.restaurant_count == 1:
                # Exactly one restaurant - either lunch or dinner
                model.Add(lunch_visit + dinner_visit == 1)
            elif self.restaurant_count >= 2:
                # Two or more restaurants - must have both lunch and dinner
                model.Add(lunch_visit == 1)
                model.Add(dinner_visit == 1)
            
            # Prevent multiple restaurants in the same meal period
            model.Add(sum(lunch_restaurants[i] for i in restaurant_pois) <= 1)
            model.Add(sum(dinner_restaurants[i] for i in restaurant_pois) <= 1)
            
            # ==== Phase 6: Meal break constraints ====
            # Prevent tourist attractions from overlapping with meal times when there's no restaurant
            tourist_in_lunch = {}
            tourist_in_dinner = {}
            
            for i in tourist_pois:
                # Tourist POI overlapping lunch time
                tourist_in_lunch[i] = model.NewBoolVar(f'tourist_{i}_in_lunch')
                
                # Define overlap: POI is active during lunch period
                arrival_before_lunch_end = model.NewBoolVar(f'arrival_{i}_before_lunch_end')
                model.Add(arrival_time[i] <= lunch_end).OnlyEnforceIf(arrival_before_lunch_end)
                model.Add(arrival_time[i] > lunch_end).OnlyEnforceIf(arrival_before_lunch_end.Not())
                
                departure_after_lunch_start = model.NewBoolVar(f'departure_{i}_after_lunch_start')
                model.Add(departure_time[i] >= lunch_start).OnlyEnforceIf(departure_after_lunch_start)
                model.Add(departure_time[i] < lunch_start).OnlyEnforceIf(departure_after_lunch_start.Not())
                
                model.AddBoolAnd([arrival_before_lunch_end, departure_after_lunch_start, visit[i]]).OnlyEnforceIf(tourist_in_lunch[i])
                model.AddBoolOr([arrival_before_lunch_end.Not(), departure_after_lunch_start.Not(), visit[i].Not()]).OnlyEnforceIf(tourist_in_lunch[i].Not())
                
                # Similar logic for dinner time
                tourist_in_dinner[i] = model.NewBoolVar(f'tourist_{i}_in_dinner')
                
                arrival_before_dinner_end = model.NewBoolVar(f'arrival_{i}_before_dinner_end')
                model.Add(arrival_time[i] <= dinner_end).OnlyEnforceIf(arrival_before_dinner_end)
                model.Add(arrival_time[i] > dinner_end).OnlyEnforceIf(arrival_before_dinner_end.Not())
                
                departure_after_dinner_start = model.NewBoolVar(f'departure_{i}_after_dinner_start')
                model.Add(departure_time[i] >= dinner_start).OnlyEnforceIf(departure_after_dinner_start)
                model.Add(departure_time[i] < dinner_start).OnlyEnforceIf(departure_after_dinner_start.Not())
                
                model.AddBoolAnd([arrival_before_dinner_end, departure_after_dinner_start, visit[i]]).OnlyEnforceIf(tourist_in_dinner[i])
                model.AddBoolOr([arrival_before_dinner_end.Not(), departure_after_dinner_start.Not(), visit[i].Not()]).OnlyEnforceIf(tourist_in_dinner[i].Not())
            
            # No tourist attractions during lunch if no lunch restaurant
            for i in tourist_pois:
                model.AddImplication(lunch_visit.Not(), tourist_in_lunch[i].Not())
            
            # No tourist attractions during dinner if no dinner restaurant
            for i in tourist_pois:
                model.AddImplication(dinner_visit.Not(), tourist_in_dinner[i].Not())

        # ==== Phase 7: Additional constraints ====
        # Enforce mandatory visits
        for poi_id in self.mandatory_visits:
            if poi_id in pois:
                model.Add(visit[poi_id] == 1)
        
        # Apply the tourist POI limit
        max_tourist_count = max_pois - self.restaurant_count
        if max_tourist_count > 0:
            model.Add(sum(visit[i] for i in tourist_pois) <= max_tourist_count)
        
        # Ensure total time doesn't exceed available time
        model.Add(sum(departure_time[i] - arrival_time[i] for i in pois) <= self.total_available_time)
        
        # Minimize travel time between POIs
        total_travel_time = model.NewIntVar(0, max_time, 'total_travel_time')
        travel_times = []
        
        for p in range(max_pois - 1):
            for i in pois:
                for j in pois:
                    if i != j:
                        travel_time = self.get_travel_time(i, j)
                        follows = model.NewBoolVar(f'travel_follows_{i}_{j}_{p}')
                        
                        model.AddBoolAnd([pos[i][p], pos[j][p+1]]).OnlyEnforceIf(follows)
                        model.AddBoolOr([pos[i][p].Not(), pos[j][p+1].Not()]).OnlyEnforceIf(follows.Not())
                        
                        travel_time_var = model.NewIntVar(0, max_time, f'travel_time_{i}_{j}_{p}')
                        model.Add(travel_time_var == travel_time).OnlyEnforceIf(follows)
                        model.Add(travel_time_var == 0).OnlyEnforceIf(follows.Not())
                        
                        travel_times.append(travel_time_var)
        
        model.Add(total_travel_time == sum(travel_times))
        
        # ==== Phase 8: Define the objective function ====
        # Primary objective: maximize interest
        interest_score = model.NewIntVar(0, 10 * len(pois), 'interest_score')
        interest_terms = []
        
        for i in pois:
            interest_var = model.NewIntVar(0, 10, f'interest_{i}')
            poi_interest = self.graph.nodes[i]['Interet']
            model.Add(interest_var == poi_interest).OnlyEnforceIf(visit[i])
            model.Add(interest_var == 0).OnlyEnforceIf(visit[i].Not())
            interest_terms.append(interest_var)
        
        model.Add(interest_score == sum(interest_terms))
        
        # Secondary objective: minimize travel time
        # Instead of dividing travel time by 10, we multiply interest by 10
        # This achieves the same relative weighting
        interest_score_scaled = model.NewIntVar(0, 100 * len(pois), 'interest_score_scaled')
        model.Add(interest_score_scaled == interest_score * 10)

        # Combined objective: maximize scaled interest score - travel time
        combined_objective = model.NewIntVar(-max_time, 100 * len(pois), 'combined_objective')
        model.Add(combined_objective == interest_score_scaled - total_travel_time)
        model.Maximize(combined_objective)
        
        # ==== Phase 9: Solve the model ====
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60  # Time limit
        status = solver.Solve(model)
        
        # ==== Phase 10: Extract the solution ====
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract the solution
            itinerary = []
            
            # Find the number of POIs in the solution
            n_visited = sum(solver.Value(visit[i]) for i in pois)
            print(f"Solution found with {n_visited} POIs")
            
            # Extract the POIs in order
            for p in range(max_pois):
                for i in pois:
                    if solver.Value(pos[i][p]) == 1:
                        arrival = solver.Value(arrival_time[i])
                        departure = solver.Value(departure_time[i])
                        itinerary.append((i, arrival, departure))
                        
                        # Calculate interest score contribution
                        poi_name = self.graph.nodes[i].get('Nom', f'POI {i}')
                        poi_type = self.graph.nodes[i].get('Type', 'attraction')
                        poi_interest = self.graph.nodes[i]['Interet']
                        print(f"Position {p+1}: {poi_name} ({poi_type}) - Interest: {poi_interest}/10")
            
            # Calculate metrics
            final_interest = sum(self.graph.nodes[i]['Interet'] for i, _, _ in itinerary)
            final_travel_time = 0
            
            for idx in range(len(itinerary) - 1):
                curr_poi, _, _ = itinerary[idx]
                next_poi, _, _ = itinerary[idx + 1]
                final_travel_time += self.get_travel_time(curr_poi, next_poi)
                
            print(f"Total interest score: {final_interest}")
            print(f"Total travel time: {final_travel_time} minutes")
                
            return itinerary
        else:
            print("No feasible solution found.")
            if status == cp_model.INFEASIBLE:
                print("Problem proven infeasible.")
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

    def _convert_itinerary_to_dict(self, itinerary):
        """Convert the solver's itinerary to a dictionary format for the API"""
        result = []
        
        for i, (poi_id, arrival, departure) in enumerate(itinerary):
            poi_data = self.graph.nodes[poi_id]
            
            # Format times as strings
            arrival_time = self._minutes_to_time_str(arrival + self.start_time)
            departure_time = self._minutes_to_time_str(departure + self.start_time)
            
            # Add the POI to the result
            poi_entry = {
                'id': poi_id,
                'name': poi_data.get('Nom', f'POI {poi_id}'),
                'type': poi_data.get('Type', 'attraction'),
                'start_time': arrival_time,
                'end_time': departure_time,
                'visit_duration': departure - arrival
            }
            
            # Add travel info if not the last POI
            if i < len(itinerary) - 1:
                next_poi_id, _, _ = itinerary[i+1]
                
                # Get transport mode and time
                try:
                    if 'selected_mode' in self.graph[poi_id][next_poi_id]:
                        transport_mode = self.graph[poi_id][next_poi_id]['selected_mode']
                    else:
                        transport_mode = self._select_preferred_transport_mode(poi_id, next_poi_id)[0]
                        
                    travel_time = self.get_travel_time(poi_id, next_poi_id, transport_mode)
                    
                    poi_entry['travel_to_next'] = {
                        'mode': ["walking", "public transport", "car"][transport_mode],
                        'time': travel_time
                    }
                except Exception as e:
                    print(f"Error getting travel info: {e}")
                    poi_entry['travel_to_next'] = {
                        'mode': "walking", 
                        'time': 30
                    }
                    
            result.append(poi_entry)
        
        return result