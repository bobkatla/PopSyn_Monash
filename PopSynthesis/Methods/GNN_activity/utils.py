from pyvis.network import Network
import torch
import json
import pandas as pd
import torch


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
    purpose_id_map = {i: purpose_id for i, purpose_id in enumerate(purposes_df["purpose_id"])}
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
