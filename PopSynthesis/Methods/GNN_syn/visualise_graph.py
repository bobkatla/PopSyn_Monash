import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, Tuple, List

def create_networkx_graph(graph: HeteroData) -> nx.DiGraph:
    """
    Convert a PyTorch Geometric HeteroData graph to a NetworkX graph for visualization.
    
    Args:
        graph: HeteroData graph
    
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node_type in ['people', 'location']:
        num_nodes = graph[node_type].num_nodes
        for i in range(num_nodes):
            G.add_node(f"{node_type}_{i}", type=node_type)
    
    # Add edges
    # People to people edges (household)
    if ('people', 'lives_with', 'people') in graph.edge_index_dict:
        edges = graph.edge_index_dict[('people', 'lives_with', 'people')].t().numpy()
        for src, dst in edges:
            G.add_edge(f"people_{src}", f"people_{dst}", type='lives_with')
    
    # Location to location edges
    if ('location', 'connected_to', 'location') in graph.edge_index_dict:
        edges = graph.edge_index_dict[('location', 'connected_to', 'location')].t().numpy()
        for src, dst in edges:
            G.add_edge(f"location_{src}", f"location_{dst}", type='connected_to')
    
    # People to location edges
    if ('people', 'visits', 'location') in graph.edge_index_dict:
        edges = graph.edge_index_dict[('people', 'visits', 'location')].t().numpy()
        if hasattr(graph['people', 'visits', 'location'], 'edge_attr'):
            attrs = graph['people', 'visits', 'location'].edge_attr.numpy()
            for i, (src, dst) in enumerate(edges):
                G.add_edge(f"people_{src}", f"location_{dst}", 
                          type='visits',
                          purpose=attrs[i][:5],
                          time=attrs[i][5],
                          joint=attrs[i][6])
        else:
            for src, dst in edges:
                G.add_edge(f"people_{src}", f"location_{dst}", type='visits')
    
    return G

def visualize_graph(graph: HeteroData, title: str = "Graph Visualization"):
    """
    Visualize the heterogeneous graph.
    
    Args:
        graph: HeteroData graph
        title: Title for the plot
    """
    G = create_networkx_graph(graph)
    
    plt.figure(figsize=(12, 8))
    
    # Define node colors based on type
    node_colors = []
    for node in G.nodes():
        if node.startswith('people'):
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgreen')
    
    # Define edge colors based on type
    edge_colors = []
    for edge in G.edges(data=True):
        if edge[2]['type'] == 'lives_with':
            edge_colors.append('blue')
        elif edge[2]['type'] == 'connected_to':
            edge_colors.append('green')
        else:
            edge_colors.append('red')
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, 
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=1000,
            font_size=8,
            arrows=True)
    
    plt.title(title)
    plt.show()

def visualize_predictions(
    graph: HeteroData,
    predictions: List[Dict],
    actual_connections: List[Dict] = None,
    title: str = "Predicted vs Actual Connections"
):
    """
    Visualize the predicted connections with their attributes.
    
    Args:
        graph: Base graph without people-location connections
        predictions: List of dictionaries containing predicted connections
        actual_connections: List of dictionaries containing actual connections
        title: Title for the plot
    """
    G = create_networkx_graph(graph)
    
    plt.figure(figsize=(15, 10))
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        if node.startswith('people'):
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgreen')
    
    # Edge colors and widths based on prediction probability
    edge_colors = []
    edge_widths = []
    edge_labels = {}
    
    # Add predicted edges
    for pred in predictions:
        src = f"people_{pred['person_id']-1}"
        dst = f"location_{pred['location_id']-1}"
        if src in G.nodes() and dst in G.nodes():
            G.add_edge(src, dst, 
                      type='predicted',
                      probability=pred['probability'],
                      purpose=pred['purpose'],
                      time=pred['time'],
                      joint=pred['joint'])
    
    # Add actual edges if available
    if actual_connections:
        for conn in actual_connections:
            src = f"people_{conn['person_id']-1}"
            dst = f"location_{conn['location_id']-1}"
            if src in G.nodes() and dst in G.nodes():
                G.add_edge(src, dst, 
                          type='actual',
                          purpose=conn['purpose'],
                          time=conn['time'],
                          joint=conn['joint'])
    
    # Set edge colors and widths
    for edge in G.edges(data=True):
        edge_key = (edge[0], edge[1])  # Use tuple of nodes as key
        if edge[2]['type'] == 'predicted':
            prob = edge[2]['probability']
            # Use a more visible color scheme
            edge_colors.append(plt.cm.Reds(0.3 + 0.7 * prob))  # Start from 0.3 to make even low probabilities visible
            edge_widths.append(1 + 4 * prob)  # Increased base width
            edge_labels[edge_key] = f"{edge[2]['purpose']}\n{edge[2]['time']:.1f}h\n{prob:.2f}"
        elif edge[2]['type'] == 'actual':
            edge_colors.append('green')
            edge_widths.append(2)
            edge_labels[edge_key] = f"{edge[2]['purpose']}\n{edge[2]['time']:.1f}h"
        elif edge[2]['type'] == 'lives_with':
            edge_colors.append('blue')
            edge_widths.append(1)
        else:
            edge_colors.append('gray')
            edge_widths.append(1)
    
    # Draw the graph with a better layout
    pos = nx.spring_layout(G, k=1, iterations=50)  # Increased k for better spacing
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1000,
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7,
                          arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold')
    
    # Add edge labels with better formatting
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=6,
                                bbox=dict(facecolor='white',
                                        edgecolor='none',
                                        alpha=0.7))
    
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.tight_layout()  # Adjust layout
    plt.show()
