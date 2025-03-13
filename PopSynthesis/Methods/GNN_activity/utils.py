from pyvis.network import Network
import torch
import json
import pandas as pd
import torch
from torch_geometric.data import HeteroData
# Force PyTorch to use CPU mode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def visualize_pyg_graph_with_zones(data, sample_people_df, sample_households_df, purposes_df, zones_df):
    """
    Creates an interactive PyVis visualization from a PyG HeteroData graph.
    It maps PyG node indices to real-world names (P1, H2, Work_Z3, Z1) for better readability.
    """
    net = Network(notebook=True, height="800px", width="100%", directed=True)
    
    # Define colors for different node types
    color_map = {
        "zone": "red",
        "purpose": "blue",
        "household": "green",
        "person": "purple",
    }

    # Create mappings from index to actual node names
    person_id_map = {i: p_id for i, p_id in enumerate(sample_people_df["person_id"])}
    household_id_map = {i: h_id for i, h_id in enumerate(sample_households_df["household_id"])}
    purpose_id_map = {i: f"{row['purpose_type']}_Z{row['zone_id']}" for i, row in purposes_df.iterrows()}
    zone_id_map = {i: row["zone_id"] for i, row in zones_df.iterrows()}  # Fixing Zone Names

    def get_node_name(node_type, index):
        """ Maps a PyG node index to its real-world ID for visualization. """
        if node_type == "person":
            return person_id_map.get(index, f"Person_{index}")
        elif node_type == "household":
            return household_id_map.get(index, f"Household_{index}")
        elif node_type == "purpose":
            return purpose_id_map.get(index, f"Purpose_{index}")
        elif node_type == "zone":
            return zone_id_map.get(index, f"Zone_{index}")  # Ensure correct zone names
        return f"{node_type}_{index}"

    # Store added nodes to avoid duplicates
    added_nodes = set()

    # Add nodes with actual names
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        for i in range(num_nodes):
            node_id = get_node_name(node_type, i)
            net.add_node(node_id, label=node_id, color=color_map.get(node_type, "gray"), title=node_type)
            added_nodes.add(node_id)

    # Add edges with readable node names
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index.numpy()

        for src, dst in zip(edge_index[0], edge_index[1]):
            src_id = get_node_name(src_type, src)
            dst_id = get_node_name(dst_type, dst)

            if src_id in added_nodes and dst_id in added_nodes:
                net.add_edge(src_id, dst_id, title=relation, width=1)

    # Enable physics for better layout
    net.toggle_physics(True)

    return net

# Save as JSON (structured nodes and edges)
def graph_to_dict(graph):
    data_dict = {
        "nodes": {nt: graph[nt].num_nodes for nt in graph.node_types},
        "edges": {et: graph[et].edge_index.numpy().tolist() for et in graph.edge_types},
        "attributes": {}
    }
    for et in graph.edge_types:
        if hasattr(graph[et], "duration"):
            data_dict["attributes"][et] = {
                "duration": graph[et].duration.numpy().tolist(),
                "ranking": graph[et].ranking.numpy().tolist(),
                "joint_activity": graph[et].joint_activity.numpy().tolist(),
            }
    return data_dict


def save_edges_to_csv(graph, filename):
    rows = []
    for et in graph.edge_types:
        edge_list = graph[et].edge_index.numpy().T
        for i, (src, dst) in enumerate(edge_list):
            row = {"source": src, "target": dst, "relation": et}
            if hasattr(graph[et], "duration"):
                row["duration"] = graph[et].duration[i].item()
                row["ranking"] = graph[et].ranking[i].item()
                row["joint_activity"] = graph[et].joint_activity[i].item()
            rows.append(row)
    pd.DataFrame(rows).to_csv(filename, index=False)


def save_graphs(graph, output_dir="./", output_name="graph", output_type="torch"):
    """Saves graphs in multiple formats for easy analysis."""

    if output_type == "torch":
        torch.save(graph, f"{output_dir}/{output_name}.pt")
    elif output_type == "json":
        with open(f"{output_dir}/{output_name}.json", "w") as f:
            json.dump(graph_to_dict(graph), f, indent=4)
    elif output_type == "csv":
        save_edges_to_csv(graph, f"{output_dir}/{output_name}.csv")

def construct_starting_graph_pyg(zones_df, purposes_df, sample_households_df, sample_people_df, od_matrix_df):
    """
    Constructs the initial PyTorch Geometric HeteroData graph with nodes and static edges.
    All persons start at home (connected to their home-purpose with duration=None).
    """
    data = HeteroData()

    # Map real-world IDs to graph indices
    zone_id_map = {zone_id: i for i, zone_id in enumerate(zones_df["zone_id"])}
    purpose_id_map = {p_id: i for i, p_id in enumerate(purposes_df["purpose_id"])}
    household_id_map = {hh_id: i for i, hh_id in enumerate(sample_households_df["household_id"])}
    person_id_map = {p_id: i for i, p_id in enumerate(sample_people_df["person_id"])}

    # Add Zone Nodes
    data["zone"].x = torch.arange(len(zones_df), dtype=torch.float).view(-1, 1)

    # Add Purpose Nodes (with attractiveness score)
    data["purpose"].x = torch.tensor(purposes_df["attractiveness_score"].values, dtype=torch.float).view(-1, 1)

    # Add Household Nodes
    data["household"].x = torch.tensor(sample_households_df[["household_income", "household_size"]].values, dtype=torch.float)

    # Add Person Nodes
    data["person"].x = torch.tensor(sample_people_df[["age"]].values, dtype=torch.float).view(-1, 1)

    # Define Zone-Zone Edges (OD Matrix)
    src, dst = [], []
    for _, row in od_matrix_df.iterrows():
        src.append(zone_id_map[row["origin"]])
        dst.append(zone_id_map[row["destination"]])
    data["zone", "travel", "zone"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Define Zone-Purpose Edges
    src, dst = [], []
    for _, row in purposes_df.iterrows():
        src.append(zone_id_map[row["zone_id"]])
        dst.append(purpose_id_map[row["purpose_id"]])
    data["zone", "has_purpose", "purpose"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Define Household-Zone Edges
    src, dst = [], []
    for _, row in sample_households_df.iterrows():
        src.append(household_id_map[row["household_id"]])
        dst.append(zone_id_map[row["zone_id"]])
    data["household", "located_in", "zone"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Define Person-Household Edges
    src, dst = [], []
    for _, row in sample_people_df.iterrows():
        src.append(person_id_map[row["person_id"]])
        dst.append(household_id_map[row["household_id"]])
    data["person", "belongs_to", "household"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Define Person-Person Edges (Household relationships)
    src, dst = [], []
    household_members = sample_people_df.groupby("household_id")["person_id"].apply(list).to_dict()
    for members in household_members.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                src.append(person_id_map[members[i]])
                dst.append(person_id_map[members[j]])
                src.append(person_id_map[members[j]])
                dst.append(person_id_map[members[i]])  # Bidirectional
    data["person", "related_to", "person"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Define Person-Purpose Initial Edges (All Persons Start at Home)
    src, dst, durations, rankings, joint_activities = [], [], [], [], []
    for _, row in sample_people_df.iterrows():
        household_zone = sample_households_df[sample_households_df["household_id"] == row["household_id"]]["zone_id"].values[0]
        home_purpose = f"{household_zone}_Residential"
        if home_purpose in purpose_id_map:
            src.append(person_id_map[row["person_id"]])
            dst.append(purpose_id_map[home_purpose])
            durations.append(0)  # Start duration as 0 (to be updated later)
            rankings.append(1)  # First activity of the day
            joint_activities.append(0)  # Home activities are solo

    # Store edge attributes properly initialized as PyTorch tensors
    data["person", "performs", "purpose"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["person", "performs", "purpose"].duration = torch.tensor(durations, dtype=torch.float)
    data["person", "performs", "purpose"].ranking = torch.tensor(rankings, dtype=torch.long)
    data["person", "performs", "purpose"].joint_activity = torch.tensor(joint_activities, dtype=torch.long)

    return data

def add_travel_diaries_to_graph(data, travel_diaries_df, person_id_map, purpose_id_map):
    """
    Adds travel diaries as edges to the graph, filling in duration, ranking, and joint activity.
    Ensures all attributes are correctly formatted as PyTorch tensors.
    """
    src, dst, durations, rankings, joint_activities = [], [], [], [], []

    for _, row in travel_diaries_df.iterrows():
        person_id = row["person_id"]
        zone_id = row["zone_id"]
        purpose_name = row["purpose"]
        duration = row["duration"]
        ranking = row["ranking_in_day"]
        joint_activity = row["joint_activity"] if pd.notna(row["joint_activity"]) else 0  # Default to 0

        # Convert person_id and purpose_id to indices
        if person_id in person_id_map and f"{purpose_name}_Z{zone_id}" in purpose_id_map:
            person_idx = person_id_map[person_id]
            purpose_idx = purpose_id_map[f"{purpose_name}_Z{zone_id}"]

            # Append new edge with attributes
            src.append(person_idx)
            dst.append(purpose_idx)
            durations.append(float(duration))
            rankings.append(int(ranking))
            joint_activities.append(int(joint_activity))  # Convert bool to int

    # Convert lists to PyTorch tensors
    edge_index_new = torch.tensor([src, dst], dtype=torch.long)
    durations_new = torch.tensor(durations, dtype=torch.float)
    rankings_new = torch.tensor(rankings, dtype=torch.long)
    joint_activities_new = torch.tensor(joint_activities, dtype=torch.long)

    # Concatenate new data with existing attributes
    data["person", "performs", "purpose"].edge_index = torch.cat([data["person", "performs", "purpose"].edge_index, edge_index_new], dim=1)
    data["person", "performs", "purpose"].duration = torch.cat([data["person", "performs", "purpose"].duration, durations_new])
    data["person", "performs", "purpose"].ranking = torch.cat([data["person", "performs", "purpose"].ranking, rankings_new])
    data["person", "performs", "purpose"].joint_activity = torch.cat([data["person", "performs", "purpose"].joint_activity, joint_activities_new])

    return data
