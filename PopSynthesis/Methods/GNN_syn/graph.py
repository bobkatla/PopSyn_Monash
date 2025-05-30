import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

def create_graph(people: pd.DataFrame, locations: pd.DataFrame, location_edges: pd.DataFrame, activity_df: pd.DataFrame) -> HeteroData:
    """
    Create a heterogeneous graph from the input data.
    
    Args:
        people: DataFrame containing people information
        locations: DataFrame containing location information
        location_edges: DataFrame containing location-to-location connections
        activity_df: DataFrame containing people-to-location activities
    
    Returns:
        HeteroData: A heterogeneous graph with people and location nodes and their connections
    """
    # Initialize the heterogeneous graph
    graph = HeteroData()
    
    # Add people nodes with indices as features
    num_people = len(people)
    graph['people'].num_nodes = num_people
    graph['people'].x = torch.arange(num_people, dtype=torch.long)  # Use indices as features
    
    # Add location nodes with attributes
    num_locations = len(locations)
    location_attrs = torch.tensor(locations['attractiveness'].values, dtype=torch.float).view(-1, 1)
    graph['location'].x = location_attrs
    
    # Create people-to-people edges (household connections)
    household_edges = []
    for hh_id in people['household_id'].unique():
        hh_members = people[people['household_id'] == hh_id]['person_id'].values
        for i in range(len(hh_members)):
            for j in range(i + 1, len(hh_members)):
                household_edges.append([hh_members[i] - 1, hh_members[j] - 1])  # -1 for 0-based indexing
    
    if household_edges:
        household_edges = torch.tensor(household_edges, dtype=torch.long).t()
        graph['people', 'lives_with', 'people'].edge_index = household_edges
        # Add reverse edges
        graph['people', 'lives_with', 'people'].edge_index = torch.cat([
            household_edges,
            household_edges.flip(0)
        ], dim=1)
    
    # Create location-to-location edges with attributes
    loc_edges = torch.tensor([
        location_edges['location_id_1'].values - 1,  # -1 for 0-based indexing
        location_edges['location_id_2'].values - 1
    ], dtype=torch.long)
    
    loc_edge_attrs = torch.tensor([
        location_edges['distance'].values,
        location_edges['travel_time'].values
    ], dtype=torch.float).t()
    
    # Add forward and reverse edges
    graph['location', 'connected_to', 'location'].edge_index = torch.cat([
        loc_edges,
        loc_edges.flip(0)
    ], dim=1)
    
    # Duplicate edge attributes for reverse edges
    graph['location', 'connected_to', 'location'].edge_attr = torch.cat([
        loc_edge_attrs,
        loc_edge_attrs
    ], dim=0)
    
    # Create people-to-location edges from activities
    activity_edges = []
    activity_attrs = []
    
    for _, row in activity_df.iterrows():
        person_idx = row['person_id_in_sequence'] - 1  # -1 for 0-based indexing
        loc_idx = row['destination_location_id'] - 1
        
        # Create edge
        activity_edges.append([person_idx, loc_idx])
        
        # Create edge attributes (one-hot encode categorical variables)
        purpose = row['purpose']
        start_time = row['start_time']
        joint = 1.0 if row['joint_activity'] else 0.0
        
        # One-hot encode purpose
        purpose_map = {
            'work': 0, 'shopping': 1, 'home': 2, 'school': 3, 'leisure': 4
        }
        purpose_one_hot = np.zeros(5)
        purpose_one_hot[purpose_map[purpose]] = 1
        
        # Combine all attributes
        attrs = np.concatenate([
            purpose_one_hot,
            [start_time / 24.0],  # Normalize time to [0,1]
            [joint]
        ])
        activity_attrs.append(attrs)
    
    if activity_edges:
        activity_edges = torch.tensor(activity_edges, dtype=torch.long).t()
        activity_attrs = torch.tensor(activity_attrs, dtype=torch.float)
        
        # Add forward and reverse edges
        graph['people', 'visits', 'location'].edge_index = torch.cat([
            activity_edges,
            activity_edges.flip(0)
        ], dim=1)
        
        # Duplicate edge attributes for reverse edges
        graph['people', 'visits', 'location'].edge_attr = torch.cat([
            activity_attrs,
            activity_attrs
        ], dim=0)
        
        # Add reverse edge type
        graph['location', 'visited_by', 'people'].edge_index = activity_edges.flip(0)
        graph['location', 'visited_by', 'people'].edge_attr = activity_attrs
    
    return graph

def create_prediction_graph(people: pd.DataFrame, locations: pd.DataFrame, location_edges: pd.DataFrame) -> HeteroData:
    """
    Create a heterogeneous graph for prediction, containing only people and location nodes
    and their existing connections, without people-to-location edges.
    
    Args:
        people: DataFrame containing people information
        locations: DataFrame containing location information
        location_edges: DataFrame containing location-to-location connections
    
    Returns:
        HeteroData: A heterogeneous graph ready for prediction
    """
    # Initialize the heterogeneous graph
    graph = HeteroData()
    
    # Add people nodes with indices as features
    num_people = len(people)
    graph['people'].num_nodes = num_people
    graph['people'].x = torch.arange(num_people, dtype=torch.long)  # Use indices as features
    
    # Add location nodes with attributes
    num_locations = len(locations)
    location_attrs = torch.tensor(locations['attractiveness'].values, dtype=torch.float).view(-1, 1)
    graph['location'].x = location_attrs
    
    # Create people-to-people edges (household connections)
    household_edges = []
    for hh_id in people['household_id'].unique():
        hh_members = people[people['household_id'] == hh_id]['person_id'].values
        for i in range(len(hh_members)):
            for j in range(i + 1, len(hh_members)):
                household_edges.append([hh_members[i] - 1, hh_members[j] - 1])  # -1 for 0-based indexing
    
    if household_edges:
        household_edges = torch.tensor(household_edges, dtype=torch.long).t()
        # Add forward and reverse edges
        graph['people', 'lives_with', 'people'].edge_index = torch.cat([
            household_edges,
            household_edges.flip(0)
        ], dim=1)
    
    # Create location-to-location edges with attributes
    loc_edges = torch.tensor([
        location_edges['location_id_1'].values - 1,  # -1 for 0-based indexing
        location_edges['location_id_2'].values - 1
    ], dtype=torch.long)
    
    loc_edge_attrs = torch.tensor([
        location_edges['distance'].values,
        location_edges['travel_time'].values
    ], dtype=torch.float).t()
    
    # Add forward and reverse edges
    graph['location', 'connected_to', 'location'].edge_index = torch.cat([
        loc_edges,
        loc_edges.flip(0)
    ], dim=1)
    
    # Duplicate edge attributes for reverse edges
    graph['location', 'connected_to', 'location'].edge_attr = torch.cat([
        loc_edge_attrs,
        loc_edge_attrs
    ], dim=0)
    
    return graph

