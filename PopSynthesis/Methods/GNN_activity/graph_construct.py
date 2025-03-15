import torch
from torch_geometric.data import HeteroData
# Force PyTorch to use CPU mode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def assign_start_time_to_travel_diaries(travel_diaries_df):

    # Ensure travel diaries are sorted by person_id and timestamp
    travel_diaries_df = travel_diaries_df.sort_values(by=["person_id", "ranking_in_day"])

    cumsum_values = travel_diaries_df.groupby("person_id")["duration"].cumsum()
    start_t = 0
    start_times = []
    for val in cumsum_values:
        start_times.append(start_t)
        if val == 1440:
            start_t = 0
        else:
            start_t = val
    travel_diaries_df["start_time"] = start_times
    travel_diaries_df["joint_activity"] = travel_diaries_df["joint_activity"].fillna(0)

    return travel_diaries_df


import torch
from torch_geometric.data import HeteroData

def construct_graph_pyg(zones_df, purposes_df, households_df, people_df, od_matrix_df, travel_diaries_df, relationship_mapping):
    data = HeteroData()

    # ID Mappings
    zone_id_map = {zone_id: i for i, zone_id in enumerate(zones_df["zone_id"])}
    purpose_id_map = {p_id: i for i, p_id in enumerate(purposes_df["purpose_id"])}
    household_id_map = {hh_id: i for i, hh_id in enumerate(households_df["household_id"])}
    person_id_map = {p_id: i for i, p_id in enumerate(people_df["person_id"])}

    # Zone Nodes
    data["zone"].x = torch.arange(len(zones_df), dtype=torch.float).view(-1, 1)

    # Purpose Nodes
    data["purpose"].x = torch.tensor(purposes_df["attractiveness_score"].values, dtype=torch.float).view(-1, 1)

    # Household Nodes
    data["household"].x = torch.tensor(households_df[["household_income", "household_size"]].values, dtype=torch.float)

    # Person Nodes
    employment_status_mapping = {'Employed': 0, 'Student': 1}
    people_df['employment_status_encoded'] = people_df['employment_status'].map(employment_status_mapping)
    data["person"].x = torch.tensor(
        people_df[["age", "employment_status_encoded"]].values, dtype=torch.float
    )

    # Zone-Zone (OD) Edges
    src_zone = [zone_id_map[row["origin"]] for _, row in od_matrix_df.iterrows()]
    dst_zone = [zone_id_map[row["destination"]] for _, row in od_matrix_df.iterrows()]
    data["zone", "connects", "zone"].edge_index = torch.tensor([src_zone, dst_zone], dtype=torch.long)
    data["zone", "connects", "zone"].edge_attr = torch.tensor(
        od_matrix_df[["distance_km", "travel_time_min"]].values, dtype=torch.float
    )
    data["zone", "connects", "zone"].t = torch.zeros(len(src_zone), dtype=torch.float)

    # Zone -> Household
    data["zone", "has_household", "household"].edge_index = torch.tensor([
        [zone_id_map[row["zone_id"]] for _, row in households_df.iterrows()],
        [household_id_map[row["household_id"]] for _, row in households_df.iterrows()]
    ], dtype=torch.long)
    data["zone", "has_household", "household"].t = torch.zeros(len(households_df), dtype=torch.float)

    # Household -> Person
    data["household", "has_person", "person"].edge_index = torch.tensor([
        [household_id_map[row["household_id"]] for _, row in people_df.iterrows()],
        [person_id_map[row["person_id"]] for _, row in people_df.iterrows()]
    ], dtype=torch.long)
    data["household", "has_person", "person"].t = torch.zeros(len(people_df), dtype=torch.float)

    # Zone -> Purpose
    data["zone", "has_purpose", "purpose"].edge_index = torch.tensor([
        [zone_id_map[row["zone_id"]] for _, row in purposes_df.iterrows()],
        [purpose_id_map[row["purpose_id"]] for _, row in purposes_df.iterrows()]
    ], dtype=torch.long)
    data["zone", "has_purpose", "purpose"].t = torch.zeros(len(purposes_df), dtype=torch.float)

    # **Purpose -> Person (Activity Scheduling)**
    src_purpose, dst_person, start_times, joint_activities = [], [], [], []

    if travel_diaries_df is not None:
        for _, row in travel_diaries_df.iterrows():
            person = person_id_map[row["person_id"]]
            zone = row["zone_id"]
            purpose = row["purpose"]
            purpose_id = f"P_{zone}_{purpose}"
            purpose_idx = purpose_id_map.get(purpose_id)
            start_time = row["start_time"]
            joint_activity = row["joint_activity"]

            if purpose_idx is None:
                print(f"🚨 Warning: Purpose {purpose_id} not found in purpose_id_map!")
                continue
            src_purpose.append(purpose_idx)
            dst_person.append(person)
            start_times.append(start_time)
            joint_activities.append(joint_activity)
    else:
        for _, row in people_df.iterrows():
            person = person_id_map[row["person_id"]]
            household_id = row["household_id"]
            zone = households_df[households_df["household_id"] == household_id]["zone_id"].values[0]
            purpose = "Residential" # At home
            purpose_id = f"P_{zone}_{purpose}"
            purpose_idx = purpose_id_map.get(purpose_id)
            start_time = 0
            joint_activity = 0

            if purpose_idx is None:
                print(f"🚨 Warning: Purpose {purpose_id} not found in purpose_id_map!")
                continue
            src_purpose.append(purpose_idx)
            dst_person.append(person)
            start_times.append(start_time)
            joint_activities.append(joint_activity)

    # Convert to PyTorch tensors
    data["purpose", "attracts", "person"].edge_index = torch.tensor([src_purpose, dst_person], dtype=torch.long)
    data["purpose", "attracts", "person"].t = torch.tensor(start_times, dtype=torch.float)
    data["purpose", "attracts", "person"].joint_activity = torch.tensor(joint_activities, dtype=torch.float)

    # **Person-Person Relationships**
    relationships = people_df.set_index('person_id')['role_in_household'].to_dict()
    household_groups = people_df.groupby("household_id")["person_id"].apply(list).to_dict()

    rel_src, rel_dst = {rel: [] for rel in relationship_mapping.keys()}, {rel: [] for rel in relationship_mapping.keys()}

    for members in household_groups.values():
        for src_person in members:
            for dst_person in members:
                if src_person == dst_person:
                    continue
                src_rel = relationships[src_person]
                dst_rel = relationships[dst_person]

                # Person-Person relationships logic (keep your existing logic here)
                if src_rel == "Main" and dst_rel == "Spouse":
                    rel_src["Spouse"].append(person_id_map[src_person])
                    rel_dst["Spouse"].append(person_id_map[dst_person])
                elif src_rel == "Spouse" and dst_rel == "Main":
                    rel_src["Spouse"].append(person_id_map[src_person])
                    rel_dst["Spouse"].append(person_id_map[dst_person])
                elif src_rel == "Main" and dst_rel == "Child":
                    rel_src["Parent"].append(person_id_map[src_person])
                    rel_dst["Parent"].append(person_id_map[dst_person])
                elif src_rel == "Spouse" and dst_rel == "Child":
                    rel_src["Parent"].append(person_id_map[src_person])
                    rel_dst["Parent"].append(person_id_map[dst_person])
                elif src_rel == "Child" and dst_rel == "Main":
                    rel_src["Child"].append(person_id_map[src_person])
                    rel_dst["Child"].append(person_id_map[dst_person])
                elif src_rel == "Child" and dst_rel == "Spouse":
                    rel_src["Child"].append(person_id_map[src_person])
                    rel_dst["Child"].append(person_id_map[dst_person])
                elif src_rel == "Housemate" and dst_rel == "Housemate":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Main" and dst_rel == "Housemate":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Housemate" and dst_rel == "Main":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Spouse" and dst_rel == "Housemate":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Housemate" and dst_rel == "Spouse":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Child" and dst_rel == "Housemate":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Housemate" and dst_rel == "Child":
                    rel_src["Housemate"].append(person_id_map[src_person])
                    rel_dst["Housemate"].append(person_id_map[dst_person])
                elif src_rel == "Child" and dst_rel == "Child":
                    rel_src["Sibling"].append(person_id_map[src_person])
                    rel_dst["Sibling"].append(person_id_map[dst_person])
                elif src_rel == "Spouse" and dst_rel == "Spouse":
                    rel_src["Spouse"].append(person_id_map[src_person])
                    rel_dst["Spouse"].append(person_id_map[dst_person])

    first_run = True
    for rel, src_nodes in rel_src.items():
        if src_nodes:  # If there are any edges of this type
            target_edges = torch.tensor([src_nodes, rel_dst[rel]], dtype=torch.long)
            target_rela = torch.tensor([relationship_mapping[rel]] * len(src_nodes), dtype=torch.long)
            if first_run:
                data["person", "relate", "person"].edge_index = target_edges
                data["person", "relate", "person"].relationship = target_rela
                data["person", "relate", "person"].t = torch.zeros(len(src_nodes), dtype=torch.float)
                first_run = False
            else:
                data["person", "relate", "person"].edge_index = torch.cat(
                    [data["person", "relate", "person"].edge_index, target_edges], dim=1)
                data["person", "relate", "person"].relationship = torch.cat(
                    [data["person", "relate", "person"].relationship, target_rela], dim=0)
                data["person", "relate", "person"].t = torch.cat(
                    [data["person", "relate", "person"].t, torch.zeros(len(src_nodes), dtype=torch.float)], dim=0)

    return data

