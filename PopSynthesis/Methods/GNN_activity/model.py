import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class TravelGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('person', 'belongs_to', 'household'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('household', 'located_in', 'zone'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('zone', 'has_purpose', 'purpose'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'performs', 'purpose'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'parent', 'person'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'child', 'person'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'spouse', 'person'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'housemate', 'person'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('person', 'sibling', 'person'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')

        # Output heads
        self.edge_classifier = Linear(hidden_channels, 1)
        self.duration_regressor = Linear(hidden_channels, 1)
        self.joint_classifier = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        embeddings = self.conv1(x_dict, edge_index_dict)
        embeddings = {k: F.relu(v) for k, v in embeddings.items()}
        return embeddings

    def predict_edges(self, person_emb, purpose_emb):
        combined = person_emb * purpose_emb
        return self.edge_classifier(combined).view(-1)

    def predict_duration(self, person_emb, purpose_emb):
        combined = person_emb * purpose_emb
        return self.duration_regressor(combined).view(-1)

    def predict_joint(self, person_emb, purpose_emb):
        combined = person_emb * purpose_emb
        return self.joint_classifier(combined).view(-1)


def train_model(model, data, epochs=50, lr=0.005, max_duration=1440):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion_edge = torch.nn.BCEWithLogitsLoss()
    criterion_duration = torch.nn.MSELoss()
    criterion_joint = torch.nn.BCEWithLogitsLoss()

    data = data.to('cpu')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data.x_dict, data.edge_index_dict)
        person_emb = embeddings["person"]
        purpose_emb = embeddings["purpose"]

        # Positive edges
        edge_index = data["person", "performs", "purpose"].edge_index
        pos_person_emb = person_emb[edge_index[0]]
        pos_purpose_emb = purpose_emb[edge_index[1]]

        # Negative Sampling
        num_neg = edge_index.size(1)
        neg_person_idx = torch.randint(0, person_emb.size(0), (num_neg,))
        neg_purpose_idx = torch.randint(0, purpose_emb.size(0), (num_neg,))
        neg_person_emb = person_emb[neg_person_idx]
        neg_purpose_emb = purpose_emb[neg_purpose_idx]

        # Predictions
        pos_edge_preds = model.predict_edges(pos_person_emb, pos_purpose_emb)
        neg_edge_preds = model.predict_edges(neg_person_emb, neg_purpose_emb)

        duration_preds = model.predict_duration(pos_person_emb, pos_purpose_emb)
        joint_preds = model.predict_joint(pos_person_emb, pos_purpose_emb)

        # Labels (normalized)
        duration_labels = data["person", "performs", "purpose"].duration / max_duration
        edge_labels = torch.cat([torch.ones_like(pos_edge_preds), torch.zeros_like(neg_edge_preds)])
        edge_preds = torch.cat([pos_edge_preds, neg_edge_preds])

        joint_labels = data["person", "performs", "purpose"].joint_activity.float()

        # Losses
        loss_edge = criterion_edge(edge_preds, edge_labels)
        loss_duration = criterion_duration(duration_preds, duration_labels)
        loss_joint = criterion_joint(joint_preds, joint_labels)

        loss = loss_edge + loss_duration + loss_joint
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss={loss.item():.4f}, "
                  f"Edge={loss_edge.item():.4f}, Duration={loss_duration.item():.4f}, Joint={loss_joint.item():.4f}")

    return model
