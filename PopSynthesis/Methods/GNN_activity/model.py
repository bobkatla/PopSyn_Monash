import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

def convert_to_temporal_data(data):
    """
    Converts PyG HeteroData to TemporalData for TGN training.
    Includes all timestamped edges instead of only 'purpose -> person'.
    """
    src_list, dst_list, t_list, msg_list = [], [], [], []

    # Purpose -> Person (Activity Data)
    src_list.append(data["purpose", "attracts", "person"].edge_index[0])
    dst_list.append(data["purpose", "attracts", "person"].edge_index[1])
    t_list.append(data["purpose", "attracts", "person"].t)  # Fixed
    msg_list.append(torch.cat([
        data["purpose", "attracts", "person"].joint_activity.view(-1, 1),
    ], dim=1))

    # Person -> Person (Social Connections)
    src_list.append(data["person", "relate", "person"].edge_index[0])
    dst_list.append(data["person", "relate", "person"].edge_index[1])
    t_list.append(data["person", "relate", "person"].t)  # Fixed
    msg_list.append(data["person", "relate", "person"].relationship.view(-1, 1))

    # Household -> Person (Household Structure)
    src_list.append(data["household", "has_person", "person"].edge_index[0])
    dst_list.append(data["household", "has_person", "person"].edge_index[1])
    t_list.append(data["household", "has_person", "person"].t)  # Fixed
    msg_list.append(torch.zeros((len(src_list[-1]), 1)))  # No extra feature, just connection

    # Zone -> Household (Location Context)
    src_list.append(data["zone", "has_household", "household"].edge_index[0])
    dst_list.append(data["zone", "has_household", "household"].edge_index[1])
    t_list.append(data["zone", "has_household", "household"].t)  # Fixed
    msg_list.append(torch.zeros((len(src_list[-1]), 1)))  # No extra feature, just connection

    # Zone -> Zone (Travel Paths)
    src_list.append(data["zone", "connects", "zone"].edge_index[0])
    dst_list.append(data["zone", "connects", "zone"].edge_index[1])
    t_list.append(data["zone", "connects", "zone"].t)  # Fixed
    msg_list.append(data["zone", "connects", "zone"].edge_attr[:, 0].view(-1, 1))  # Pick only `distance_km`

    # Stack everything into TemporalData
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    t = torch.cat(t_list)
    msg = torch.cat(msg_list)

    return TemporalData(src=src, dst=dst, t=t, msg=msg)

# === Graph Attention Embedding ===
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t  # Relative time encoding
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


# === Link Predictor (Activity Prediction) ===
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)  # Binary classification for edge existence
        self.lin_joint = Linear(in_channels, 1)  # Binary classification for joint activity

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        edge_pred = self.lin_final(h)  # Edge presence probability
        joint_pred = self.lin_joint(h)  # Joint activity probability
        return edge_pred, joint_pred

