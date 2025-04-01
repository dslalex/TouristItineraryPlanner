import networkx as nx
import pickle
import itertools
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_graph_path(city):
    """Return the path for a city's graph file."""
    city = city.lower().replace(" ", "_")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(data_dir, f"{city}_graph.pkl")

def create_graph(locations, city="paris"):
    """Create a graph from a list of locations."""
    G = nx.Graph()
    
    # Add nodes
    for loc in locations:
        G.add_node(loc["ID"], **loc)

    # Add edges without travel times - they will be calculated on demand
    for u, v in itertools.combinations(G.nodes(), 2):
        G.add_edge(u, v)
    
    return G

def save_graph(G, city="paris"):
    """Save a city graph to disk."""
    file_path = get_graph_path(city)
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {file_path}")

def load_graph(city="paris"):
    """Load a city graph from disk."""
    file_path = get_graph_path(city)
    
    # Special case for Paris
    if city.lower() == "paris":
        try:
            # Try to create the Paris graph directly
            from data.paris_graph import create_graph as create_paris_graph
            from data.paris_graph import save_graph as save_paris_graph
            G = create_paris_graph()
            # Save it if not already saved
            if not os.path.exists(file_path):
                save_graph(G, city)
            return G
        except Exception as e:
            print(f"Error loading Paris graph: {e}")
            # Continue to try loading from file
    
    # Check if file exists
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded for {city}")
    return G

def display_graph_window(G, city="Unknown City"):
    """Display a visualization of the city graph."""
    pos = {n: (G.nodes[n]["longitude"], G.nodes[n]["latitude"]) for n in G.nodes()}
    plt.figure(figsize=(10, 8))
    
    # Extract node attributes for visualization
    node_types = [G.nodes[n].get("Type", "Unknown") for n in G.nodes()]
    node_interest = [G.nodes[n].get("Interet", 5) for n in G.nodes()]
    node_sizes = [max(30, n * 10) for n in node_interest]  # Scale interest to node size
    
    # Color nodes by type
    color_map = {"Touristique": "blue", "Restaurant": "red", "Unknown": "gray"}
    node_colors = [color_map.get(t, "gray") for t in node_types]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    
    # Add labels for important nodes
    labels = {n: G.nodes[n]["Nom"] for n in G.nodes() if n <= 50}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add legend
    for type_name, color in color_map.items():
        plt.plot([], [], 'o', color=color, label=type_name)
    plt.legend(loc="best")
    
    plt.title(f"{city} Locations Graph")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.show()

# Import Paris data for backward compatibility
# from data.paris_graph import known_locations as paris_locations

# def get_paris_graph():
#     """Legacy function to maintain compatibility with existing code."""
#     return create_graph(paris_locations, "paris")