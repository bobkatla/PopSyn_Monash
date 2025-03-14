import torch
from torch_geometric.data import HeteroData
# Force PyTorch to use CPU mode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def construct_starting_graph_pyg(zones_df, purposes_df, sample_households_df, sample_people_df, od_matrix_df):
    data = HeteroData()

    # ID Mapping
    zone_id_map = {zone_id: i for i, zone_id in enumerate(zones_df["zone_id"])}
    purpose_id_map = {p_id: i for i, p_id in enumerate(purposes_df["purpose_id"])}
    household_id_map = {hh_id: i for i, hh_id in enumerate(sample_households_df["household_id"])}
    person_id_map = {p_id: i for i, p_id in enumerate(sample_people_df["person_id"])}

    # Add Zone Nodes
    data["zone"].x = torch.arange(len(zones_df), dtype=torch.float).view(-1, 1)
    # Add Purpose Nodes
    data["purpose"].x = torch.tensor(purposes_df["attractiveness_score"].values, dtype=torch.float).view(-1, 1)
    # Add Household Nodes
    data["household"].x = torch.tensor(sample_households_df[["household_income", "household_size"]].values, dtype=torch.float)

    # Define employment status encoding (only 'Employed' and 'Student')
    employment_status_mapping = {'Employed': 0, 'Student': 1}
    # Encode employment status as numeric
    sample_people_df['employment_status_encoded'] = sample_people_df['employment_status'].map(employment_status_mapping)
    # Add Person Nodes (features: age and employment_status)
    data["person"].x = torch.tensor(
        sample_people_df[["age", "employment_status_encoded"]].values,
        dtype=torch.float
    )

    # Zone-Zone (OD) Edges
    data["zone", "travel", "zone"].edge_index = torch.tensor([
        [zone_id_map[row["origin"]] for _, row in od_matrix_df.iterrows()],
        [zone_id_map[row["destination"]] for _, row in od_matrix_df.iterrows()]
    ], dtype=torch.long)

    # Zone-Purpose Edges
    data["zone", "has_purpose", "purpose"].edge_index = torch.tensor([
        [zone_id_map[row["zone_id"]] for _, row in purposes_df.iterrows()],
        [purpose_id_map[row["purpose_id"]] for _, row in purposes_df.iterrows()]
    ], dtype=torch.long)

    # Household-Zone Edges
    data["household", "located_in", "zone"].edge_index = torch.tensor([
        [household_id_map[row["household_id"]] for _, row in sample_households_df.iterrows()],
        [zone_id_map[row["zone_id"]] for _, row in sample_households_df.iterrows()]
    ], dtype=torch.long)

    # Person-Household Edges
    data["person", "belongs_to", "household"].edge_index = torch.tensor([
        [person_id_map[row["person_id"]] for _, row in sample_people_df.iterrows()],
        [household_id_map[row["household_id"]] for _, row in sample_people_df.iterrows()]
    ], dtype=torch.long)

    # Person-Person Relationship Edges
    relationships = sample_people_df.set_index('person_id')['role_in_household'].to_dict()
    household_groups = sample_people_df.groupby("household_id")["person_id"].apply(list).to_dict()

    rel_src, rel_dst = {rel: [] for rel in ["Main", "Spouse", "Child", "Parent", "Housemate", "Sibling"]}, \
                       {rel: [] for rel in ["Main", "Spouse", "Child", "Parent", "Housemate", "Sibling"]}

    for members in household_groups.values():
        for src_person in members:
            for dst_person in members:
                if src_person == dst_person:
                    continue
                src_rel = relationships[src_person]
                dst_rel = relationships[dst_person]

                # Define edge directions and labels
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

    for rel, src_nodes in rel_src.items():
        if src_nodes:  # If there are any edges of this type
            data["person", rel.lower(), "person"].edge_index = torch.tensor(
                [src_nodes, rel_dst[rel]], dtype=torch.long)

    # Initial Person-Purpose Edges (Starting at Home)
    src, dst, duration, joint_activity = [], [], [], []
    for _, person in sample_people_df.iterrows():
        hh_zone = sample_households_df.loc[sample_households_df["household_id"] == person["household_id"], "zone_id"].values[0]
        home_purpose_id = f"P_{hh_zone}_Residential"

        if home_purpose_id in purpose_id_map:
            src.append(person_id_map[person["person_id"]])
            dst.append(purpose_id_map[home_purpose_id])
            duration.append(0.0)
            joint_activity.append(0)

    data["person", "performs", "purpose"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["person", "performs", "purpose"].duration = torch.tensor(duration, dtype=torch.float)
    data["person", "performs", "purpose"].joint_activity = torch.tensor(joint_activity, dtype=torch.long)

    return data

def add_travel_diaries_to_graph(data, travel_diaries_df, person_id_map, purpose_id_map):
    """
    Adds person → purpose edges based on travel diaries, with attributes for duration, ranking, and joint activity.
    All activities start from midnight and duration is given in minutes.
    """
    src, dst, durations, rankings, joint_activities = [], [], [], [], []

    for person_id, person_activities in travel_diaries_df.groupby("person_id"):
        person = person_id_map[person_id]

        # Sort each person's activities by ranking
        person_activities_sorted = person_activities.sort_values("ranking_in_day")

        for _, row in person_activities_sorted.iterrows():
            zone = row["zone_id"]
            purpose = row["purpose"]
            purpose_id = f"P_{zone}_{purpose}"
            purpose_idx = purpose_id_map.get(purpose_id)

            if purpose_idx is None:
                print(f"🚨 Purpose {purpose_id} not found in purpose_id_map!")
                continue

            src.append(person)
            dst.append(purpose_idx)
            durations.append(row["duration"])
            rankings.append(row["ranking_in_day"])
            joint_activities.append(1 if row["joint_activity"] else 0)

    if len(src) == 0:
        print("🚨 No person → purpose edges found in travel diaries!")
        data["person", "performs", "purpose"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["person", "performs", "purpose"].duration = torch.zeros((0,), dtype=torch.float32)
        data["person", "performs", "purpose"].ranking = torch.zeros((0,), dtype=torch.long)
        data["person", "performs", "purpose"].joint_activity = torch.zeros((0,), dtype=torch.long)
        return data

    data["person", "performs", "purpose"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["person", "performs", "purpose"].duration = torch.tensor(durations, dtype=torch.float32)
    data["person", "performs", "purpose"].ranking = torch.tensor(rankings, dtype=torch.long)
    data["person", "performs", "purpose"].joint_activity = torch.tensor(joint_activities, dtype=torch.long)

    return data
