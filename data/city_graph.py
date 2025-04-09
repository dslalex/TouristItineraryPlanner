import networkx as nx
import pickle
import itertools
import os
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import import_module
import logging
import traceback

logger = logging.getLogger(__name__)

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

def check_city_graph_exists(city):
    """Check if a graph for the given city already exists"""
    graph_path_graphml = os.path.join(os.path.dirname(__file__), f"{city}.graphml")
    graph_path_pkl = os.path.join(os.path.dirname(__file__), f"{city}_graph.pkl")
    
    exists_graphml = os.path.exists(graph_path_graphml)
    exists_pkl = os.path.exists(graph_path_pkl)
    
    logger.debug(f"Graph existence check for {city}: graphml={exists_graphml}, pkl={exists_pkl}")
    
    return exists_graphml or exists_pkl

def load_graph(city):
    """Load city graph and return it"""
    logger.info(f"Loading graph for {city}")
    
    # First try to load from .graphml file
    graph_path_graphml = os.path.join(os.path.dirname(__file__), f"{city}.graphml")
    graph_path_pkl = os.path.join(os.path.dirname(__file__), f"{city}_graph.pkl")
    
    # Check for .graphml file
    if os.path.exists(graph_path_graphml):
        try:
            logger.info(f"Loading {city} graph from file: {graph_path_graphml}")
            return nx.read_graphml(graph_path_graphml)
        except Exception as e:
            logger.error(f"Error loading {city} graph from graphml file: {str(e)}")
    
    # Check for .pkl file
    if os.path.exists(graph_path_pkl):
        try:
            logger.info(f"Loading {city} graph from pickle file: {graph_path_pkl}")
            with open(graph_path_pkl, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {city} graph from pickle file: {str(e)}")
    
    # If file doesn't exist, try all possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), f"{city}.pkl"),
        os.path.join(os.path.dirname(__file__), f"{city.lower()}_graph.pkl"),
        os.path.join(os.path.dirname(__file__), f"{city.lower()}.pkl"),
        os.path.join(os.path.dirname(__file__), f"{city}.pickle"),
        os.path.join(os.path.dirname(__file__), f"{city.lower()}.pickle")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading {city} graph from alternate file: {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading from {path}: {str(e)}")
    
    # Last resort, try to import from module
    try:
        module_name = f"data.{city}_graph"
        logger.info(f"Attempting to import {module_name}")
        
        module = import_module(module_name)
        
        if hasattr(module, 'get_graph'):
            logger.info(f"Using get_graph() from {module_name}")
            return module.get_graph()
        elif hasattr(module, 'create_graph'):
            logger.info(f"Using create_graph() from {module_name}")
            return module.create_graph()
        else:
            logger.warning(f"No graph creation function found in {module_name}")
            return None
            
    except ImportError as e:
        logger.error(f"Error loading {city} graph: {str(e)}")
        return None

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