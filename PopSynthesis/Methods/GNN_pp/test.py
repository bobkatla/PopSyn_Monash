import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

class HouseholdDiffusionGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        household_feature_dim,
        hidden_dim=128,
        num_diffusion_steps=1000,
        min_beta=1e-4,
        max_beta=0.02,
        gnn_layers=3,
        use_attention=True
    ):
        super(HouseholdDiffusionGNN, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.household_feature_dim = household_feature_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.use_attention = use_attention
        
        # Define diffusion schedule (linear beta schedule)
        self.betas = torch.linspace(min_beta, max_beta, num_diffusion_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.fc_projection = torch.nn.Linear(hidden_dim, node_feature_dim + hidden_dim * 2)  # Project to 266 after first GNN layer
        
        # Household feature embedding
        self.household_encoder = nn.Sequential(
            nn.Linear(household_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers for node feature denoising
        if use_attention:
            self.gnn_layers = nn.ModuleList([
                GATv2Conv(
                    in_channels=node_feature_dim + hidden_dim * 2, 
                    out_channels=hidden_dim,
                    edge_dim=edge_feature_dim
                ) for _ in range(gnn_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                GCNConv(
                    in_channels=node_feature_dim + hidden_dim * 2, 
                    out_channels=hidden_dim
                ) for _ in range(gnn_layers)
            ])
            
        # Output projection for node features
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )
        
        # Edge feature prediction network
        self.edge_prediction = nn.Sequential(
            nn.Linear(node_feature_dim * 2 + household_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_feature_dim)
        )
        
    def add_noise(self, x, t):
        """Add noise to the clean data x according to diffusion schedule at step t"""
        # Extract appropriate alphas for the batch of timesteps
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        return noisy_x, noise
        
    def denoise_step(self, x_t, t, household_features, edge_index, edge_attr=None, batch=None):
        """Single denoising step: predict noise to remove from x_t"""
        # Embed time step t
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Process household features
        if batch is not None:
            unique_batches = torch.unique(batch)
            h_embs = []
            
            for b_idx in range(len(unique_batches)):
                b = unique_batches[b_idx]
                h_feat = household_features[b_idx:b_idx+1]  # Adjust indexing
                h_emb = self.household_encoder(h_feat)
                batch_mask = (batch == b)
                num_nodes_in_batch = batch_mask.sum()
                h_embs.append(h_emb.repeat(num_nodes_in_batch, 1))
            
            h_emb = torch.cat(h_embs, dim=0)
        else:
            h_emb = self.household_encoder(household_features)
            h_emb = h_emb.repeat(x_t.size(0), 1)
        
        if len(t_emb) == 1:
            t_emb = t_emb.repeat(x_t.size(0), 1)
        
        # Concatenate node features with time and household embeddings
        x = torch.cat([x_t, t_emb, h_emb], dim=1)
        
        # Process through GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.use_attention and edge_attr is not None:
                x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = gnn_layer(x, edge_index)
            x = F.silu(x)
            if i < len(self.gnn_layers) - 1:
                x = self.fc_projection(x)
        
        # Project to get predicted noise
        pred_noise = self.node_projection(x)
        
        return pred_noise

    
    def predict_edge_features(self, node_features, household_features, edge_index, batch=None):
        """Predict edge features based on connected node features and household attributes"""
        # Get node features for both ends of each edge
        src, dst = edge_index
        node_features_src = node_features[src]
        node_features_dst = node_features[dst]
        
        # Get household features for each edge
        if batch is not None:
            # For each edge, get the household features of its source node
            edge_household_features = []
            for s in src:
                # Get batch index for this node
                b_idx = batch[s].item()
                # Get household features
                edge_household_features.append(household_features[b_idx])
            edge_household_features = torch.stack(edge_household_features)
        else:
            # Single household case - use the same household features for all edges
            edge_household_features = household_features.repeat(edge_index.size(1), 1)
            
        # Concatenate node features from both ends with household features
        edge_inputs = torch.cat([node_features_src, node_features_dst, edge_household_features], dim=1)
        
        # Predict edge features
        pred_edge_features = self.edge_prediction(edge_inputs)
        
        return pred_edge_features
        
    def forward(self, x_t, t, household_features, edge_index, edge_attr=None, batch=None):
        """Forward pass: predict the noise in x_t"""
        return self.denoise_step(x_t, t, household_features, edge_index, edge_attr, batch)
    
    def sample(self, household_features, num_nodes_per_household, device="cpu"):
        """Sample node features using the diffusion model for given household features"""
        # Determine batch size from household_features
        batch_size = household_features.size(0)
        
        # Initialize with random noise
        x = torch.randn(batch_size * num_nodes_per_household, self.node_feature_dim, device=device)
        
        # Create edge indices for fully connected graphs within each household
        edge_indices = []
        batch_indices = []
        
        # For each household in the batch
        for i in range(batch_size):
            # Generate fully connected graph for this household
            for j in range(num_nodes_per_household):
                for k in range(num_nodes_per_household):
                    if j != k:  # Connect all nodes except self-loops
                        src_idx = i * num_nodes_per_household + j
                        dst_idx = i * num_nodes_per_household + k
                        edge_indices.append([src_idx, dst_idx])
            
            # Create batch indices for each node
            batch_indices.extend([i] * num_nodes_per_household)
            
        edge_index = torch.tensor(edge_indices, device=device).t().contiguous()
        batch = torch.tensor(batch_indices, device=device)
        
        # Reverse diffusion process (gradually denoise)
        for t in range(self.num_diffusion_steps - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device).repeat(x.size(0))
            
            with torch.no_grad():
                # Predict noise
                pred_noise = self.denoise_step(x, t_tensor, household_features, edge_index, None, batch)
                
                # Get alpha values for current step
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                
                # Calculate coefficients
                beta = 1 - alpha
                coef1 = beta / torch.sqrt(1 - alpha_cumprod)
                coef2 = alpha * torch.sqrt(1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                
                # Update x
                if t > 0:
                    noise = torch.randn_like(x)
                    x = (x - coef1 * pred_noise) / torch.sqrt(alpha)
                    x = x + coef2 * noise
                else:
                    # Last step, no more noise to add
                    x = (x - coef1 * pred_noise) / torch.sqrt(alpha)
        
        # Predict edge features based on final node features
        pred_edge_features = self.predict_edge_features(x, household_features, edge_index, batch)
        
        # Reshape the results to get features per household
        node_features_per_household = []
        for i in range(batch_size):
            household_mask = (batch == i)
            household_nodes = x[household_mask]
            node_features_per_household.append(household_nodes)
            
        # Group edge features by household
        edge_features_per_household = []
        edge_indices_per_household = []
        
        edge_counter = 0
        for i in range(batch_size):
            household_edges = []
            household_edge_indices = []
            
            for j in range(num_nodes_per_household):
                for k in range(num_nodes_per_household):
                    if j != k:
                        household_edges.append(pred_edge_features[edge_counter])
                        household_edge_indices.append([j, k])
                        edge_counter += 1
                        
            edge_features_per_household.append(torch.stack(household_edges))
            edge_indices_per_household.append(torch.tensor(household_edge_indices).t())
            
        return node_features_per_household, edge_features_per_household, edge_indices_per_household


# Data processing functions
def prepare_household_data(household_attrs, person_features, relationships):
    """
    Prepare household data for training
    
    Args:
        household_attrs: List of household attribute tensors [num_households, household_feature_dim]
        person_features: List of lists of person features per household
                        [num_households, [num_persons_i, person_feature_dim]]
        relationships: List of lists of relationship features between persons
                      [num_households, [num_relationships_i, relationship_feature_dim]]
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    data_list = []
    
    for i, (h_attr, persons, rels) in enumerate(zip(household_attrs, person_features, relationships)):
        num_nodes = len(persons)
        
        # Create node features tensor
        x = torch.stack(persons)
        
        # Create fully connected edge index (excluding self-loops)
        edge_index = []
        for j in range(num_nodes):
            for k in range(num_nodes):
                if j != k:
                    edge_index.append([j, k])
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        # Edge attributes
        edge_attr = torch.stack(rels)
        
        # Create Data object - ensure household_attr is a proper tensor
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            household_attr=h_attr.unsqueeze(0) if h_attr.dim() == 1 else h_attr,  # Ensure it's 2D
            num_nodes=num_nodes
        )
        
        data_list.append(data)
        
    return data_list

def collate_varying_size_graphs(data_list):
    """Custom collate function for batching graphs of different sizes"""
    batch = Batch.from_data_list(data_list)
    
    # Extract household attributes - they're already in the batch as 'household_attr'
    # No need to add them again
    
    return batch

def train_household_diffusion_model(
    model, 
    train_loader,
    optimizer,
    num_epochs=100,
    device="cpu",
    log_interval=10
):
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Extract household attributes
            household_attrs = batch.household_attr
            
            # Extract node features
            x = batch.x
            
            # Generate random timesteps - one per node
            t = torch.randint(0, model.num_diffusion_steps, (x.size(0),), device=device)
            
            # Add noise to node features
            noisy_x, target_noise = model.add_noise(x, t)
            
            # Predict noise
            pred_noise = model(noisy_x, t, household_attrs, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Calculate node feature loss (MSE between predicted and target noise)
            node_loss = F.mse_loss(pred_noise, target_noise)
            
            # For edge attributes, we'll predict them from denoised node features
            clean_node_features = x
            pred_edge_attr = model.predict_edge_features(
                clean_node_features, 
                household_attrs, 
                batch.edge_index, 
                batch.batch
            )
            
            # Calculate edge attribute loss
            edge_loss = F.mse_loss(pred_edge_attr, batch.edge_attr)
            
            # Combine losses
            loss = node_loss + edge_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_epoch_loss = epoch_loss / batch_count
        losses.append(avg_epoch_loss)
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return losses

# Synthetic data generation for testing
def generate_synthetic_household_data(
    num_households=100,
    min_household_size=2,
    max_household_size=6,
    household_feature_dim=8,
    person_feature_dim=10,
    relationship_feature_dim=5
):
    """Generate synthetic household data for testing the model"""
    household_attrs = []
    person_features = []
    relationships = []
    
    for _ in range(num_households):
        # Generate household attributes
        h_attr = torch.randn(household_feature_dim)
        household_attrs.append(h_attr)
        
        # Determine household size
        household_size = torch.randint(min_household_size, max_household_size + 1, (1,)).item()
        
        # Generate person features
        persons = []
        for _ in range(household_size):
            # Person features are correlated with household attributes
            # Fix: Pad household attributes if person_feature_dim > household_feature_dim
            if person_feature_dim > household_feature_dim:
                padded_h_attr = torch.cat([h_attr, torch.zeros(person_feature_dim - household_feature_dim)])
                p_attr = torch.randn(person_feature_dim) + 0.5 * padded_h_attr
            else:
                p_attr = torch.randn(person_feature_dim) + 0.5 * h_attr[:person_feature_dim]
            persons.append(p_attr)
        person_features.append(persons)
        
        # Generate relationships between all pairs
        rels = []
        for i in range(household_size):
            for j in range(household_size):
                if i != j:
                    # Relationship features depend on the connected persons
                    # Make sure we only take the first relationship_feature_dim elements
                    r_attr = torch.randn(relationship_feature_dim) + 0.3 * (
                        persons[i][:relationship_feature_dim] + 
                        persons[j][:relationship_feature_dim]
                    )
                    rels.append(r_attr)
        relationships.append(rels)
    
    return household_attrs, person_features, relationships

# Demo usage
def run_demo():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    node_feature_dim = 10
    edge_feature_dim = 5
    household_feature_dim = 8
    hidden_dim = 128
    num_diffusion_steps = 100  # Reduced for demo purposes
    
    # Generate synthetic data
    print("Generating synthetic data...")
    household_attrs, person_features, relationships = generate_synthetic_household_data(
        num_households=100,
        person_feature_dim=node_feature_dim,
        relationship_feature_dim=edge_feature_dim,
        household_feature_dim=household_feature_dim
    )
    
    # Prepare data for training
    print("Preparing data...")
    data_list = prepare_household_data(household_attrs, person_features, relationships)
    
    # Split data
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(
        train_data, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_varying_size_graphs
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=8, 
        shuffle=False,
        collate_fn=collate_varying_size_graphs
    )
    
    # Initialize model
    print("Initializing model...")
    model = HouseholdDiffusionGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        household_feature_dim=household_feature_dim,
        hidden_dim=hidden_dim,
        num_diffusion_steps=num_diffusion_steps,
        use_attention=True
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train model
    print("Training model...")
    losses = train_household_diffusion_model(
        model, 
        train_loader,
        optimizer,
        num_epochs=60,  # Reduced for demo
        device=device,
        log_interval=5
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # Test inference
    print("Testing inference...")
    model.eval()
    
    # Choose a test household attribute
    test_household_attr = household_attrs[train_size].unsqueeze(0).to(device)
    
    # Sample features for a household with 4 members
    node_features, edge_features, edge_indices = model.sample(
        test_household_attr,
        num_nodes_per_household=4,
        device=device
    )
    
    print(f"Generated node features for household: {node_features[0].shape}")
    print(f"Generated edge features for household: {edge_features[0].shape}")
    
    # Compare with ground truth
    print("\nVisualizing feature distributions...")
    
    # Get original features for comparison
    orig_nodes = person_features[train_size]
    orig_edges = relationships[train_size]
    
    # Convert to tensor if necessary
    if not isinstance(orig_nodes, torch.Tensor):
        orig_nodes = torch.stack(orig_nodes)
    
    if not isinstance(orig_edges, torch.Tensor):
        orig_edges = torch.stack(orig_edges)
    
    # Plot distributions of node features (first dimension)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(node_features[0].cpu().detach()[:, 0], bins=20, alpha=0.5, label='Generated')
    plt.hist(orig_nodes.cpu().detach()[:, 0], bins=20, alpha=0.5, label='Original')
    plt.title('Node Feature Distribution (First Dimension)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(edge_features[0].cpu().detach()[:, 0], bins=20, alpha=0.5, label='Generated')
    plt.hist(orig_edges.cpu().detach()[:, 0], bins=20, alpha=0.5, label='Original')
    plt.title('Edge Feature Distribution (First Dimension)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    run_demo()