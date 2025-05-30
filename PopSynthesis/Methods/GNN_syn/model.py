import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, GATConv
from torch_geometric.data import HeteroData
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from visualise_graph import create_networkx_graph, visualize_graph, visualize_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int, dropout: float = 0.2):
        """
        Initialize the Heterogeneous GNN model.
        
        Args:
            hidden_channels: Number of hidden channels in the model
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Node type encoders with batch normalization
        self.person_encoder = torch.nn.Sequential(
            Linear(1, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )
        self.location_encoder = torch.nn.Sequential(
            Linear(1, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )
        
        # Edge type encoders with batch normalization
        self.household_encoder = torch.nn.Sequential(
            Linear(1, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )
        self.location_connection_encoder = torch.nn.Sequential(
            Linear(2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )
        self.activity_encoder = torch.nn.Sequential(
            Linear(7, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )
        
        # Convolutional layers with residual connections
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                ('people', 'lives_with', 'people'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
                ('location', 'connected_to', 'location'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
                ('people', 'visits', 'location'): GATConv(hidden_channels, hidden_channels, add_self_loops=False),
                ('location', 'visited_by', 'people'): GATConv(hidden_channels, hidden_channels, add_self_loops=False)
            }, aggr='mean')
            self.convs.append(conv)
            
            # Add batch normalization for each node type
            self.batch_norms.append(torch.nn.ModuleDict({
                'people': torch.nn.BatchNorm1d(hidden_channels),
                'location': torch.nn.BatchNorm1d(hidden_channels)
            }))
        
        # Prediction heads with skip connections
        self.edge_pred = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, 1)
        )
        
        self.purpose_pred = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, 5)
        )
        
        self.time_pred = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, 1)
        )
        
        self.joint_pred = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_attr_dict: Dictionary of edge attributes
        
        Returns:
            Tuple of (edge existence predictions, purpose predictions, time predictions, joint predictions,
                     person indices, location indices)
        """
        # Encode node features
        x_dict['people'] = self.person_encoder(x_dict['people'].float().view(-1, 1))
        x_dict['location'] = self.location_encoder(x_dict['location'])
        
        # Store initial embeddings for residual connections
        initial_embeddings = {key: x.clone() for key, x in x_dict.items()}
        
        # Encode edge features
        new_edge_attr_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type == ('people', 'lives_with', 'people'):
                new_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), 1, device=edge_index.device)
            elif edge_type == ('location', 'connected_to', 'location'):
                if edge_type in edge_attr_dict:
                    new_edge_attr_dict[edge_type] = self.location_connection_encoder(edge_attr_dict[edge_type])
                else:
                    new_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), 1, device=edge_index.device)
            elif edge_type == ('people', 'visits', 'location'):
                if edge_type in edge_attr_dict:
                    new_edge_attr_dict[edge_type] = self.activity_encoder(edge_attr_dict[edge_type])
                else:
                    new_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), 1, device=edge_index.device)
            else:
                new_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), 1, device=edge_index.device)
        
        # Apply convolutional layers with residual connections
        for i, (conv, batch_norms) in enumerate(zip(self.convs, self.batch_norms)):
            # Store previous embeddings for residual connection
            prev_x_dict = {key: x.clone() for key, x in x_dict.items()}
            
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict, new_edge_attr_dict)
            
            # Apply batch normalization and residual connection
            for node_type in ['people', 'location']:
                x_dict[node_type] = batch_norms[node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = self.dropout(x_dict[node_type])
                # Add residual connection
                x_dict[node_type] = x_dict[node_type] + prev_x_dict[node_type]
        
        # Get all possible people-location pairs
        num_people = x_dict['people'].size(0)
        num_locations = x_dict['location'].size(0)
        
        # Create all possible pairs
        people_idx = torch.arange(num_people, device=x_dict['people'].device)
        location_idx = torch.arange(num_locations, device=x_dict['location'].device)
        
        # Get node embeddings for all pairs
        people_emb = x_dict['people'][people_idx]
        location_emb = x_dict['location'][location_idx]
        
        # Concatenate embeddings for each pair
        pair_emb = torch.cat([people_emb, location_emb], dim=1)
        
        # Make predictions
        edge_pred = torch.sigmoid(self.edge_pred(pair_emb))
        purpose_pred = self.purpose_pred(pair_emb)
        time_pred = torch.sigmoid(self.time_pred(pair_emb)) * 24.0  # Scale to [0, 24]
        joint_pred = torch.sigmoid(self.joint_pred(pair_emb))
        
        return edge_pred, purpose_pred, time_pred, joint_pred, people_idx, location_idx

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for binary classification.
    
    Args:
        pred: Predicted probabilities
        target: Target labels
        alpha: Weighting factor for positive samples
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()

def train_model(
    model: HeteroGNN,
    train_graph: HeteroData,
    val_graph: HeteroData,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    batch_size: int = 32,
    patience: int = 20,
    min_delta: float = 1e-4
) -> Tuple[List[float], List[float]]:
    """
    Train the HeteroGNN model with early stopping.
    
    Args:
        model: The HeteroGNN model
        train_graph: Training graph
        val_graph: Validation graph
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        batch_size: Batch size for training
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in validation loss to be considered as improvement
    
    Returns:
        Tuple of training and validation losses
    """
    model = model.to(device)
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)
    
    train_losses = []
    val_losses = []
    
    # Get ground truth edges and attributes
    train_edges = train_graph['people', 'visits', 'location'].edge_index
    train_edge_attrs = train_graph['people', 'visits', 'location'].edge_attr
    val_edges = val_graph['people', 'visits', 'location'].edge_index
    val_edge_attrs = val_graph['people', 'visits', 'location'].edge_attr
    
    # Create positive and negative samples
    train_pos_edges = train_edges.t()
    val_pos_edges = val_edges.t()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Loss weights for task balancing
    edge_weight = 2.0  # Increased weight for edge prediction
    purpose_weight = 1.0
    time_weight = 0.5
    joint_weight = 0.5
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        edge_pred, purpose_pred, time_pred, joint_pred, person_idx, location_idx = model(
            train_graph.x_dict,
            train_graph.edge_index_dict,
            train_graph.edge_attr_dict
        )
        
        # Create labels for all possible pairs
        edge_labels = torch.zeros(len(person_idx), device=device)
        purpose_labels = torch.zeros(len(person_idx), 5, device=device)
        time_labels = torch.zeros(len(person_idx), device=device)
        joint_labels = torch.zeros(len(person_idx), device=device)
        
        for i, (p, l) in enumerate(zip(person_idx, location_idx)):
            mask = (train_pos_edges[:, 0] == p) & (train_pos_edges[:, 1] == l)
            if any(mask):
                edge_labels[i] = 1
                # Get the corresponding edge attributes
                edge_idx = torch.where(mask)[0][0]
                attrs = train_edge_attrs[edge_idx]
                
                # Purpose (one-hot encoded)
                purpose_labels[i] = attrs[:5]
                # Time (normalized to [0,1])
                time_labels[i] = attrs[5]
                # Joint activity
                joint_labels[i] = attrs[6]
        
        # Calculate losses with task weights
        edge_loss = focal_loss(edge_pred.squeeze(), edge_labels) * edge_weight
        purpose_loss = F.cross_entropy(purpose_pred, purpose_labels) * purpose_weight
        time_loss = F.mse_loss(time_pred.squeeze(), time_labels) * time_weight
        joint_loss = F.binary_cross_entropy(joint_pred.squeeze(), joint_labels) * joint_weight
        
        # Add L2 regularization
        l2_lambda = 0.0001  # Further reduced L2 regularization
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        
        # Total loss
        loss = edge_loss + purpose_loss + time_loss + joint_loss + l2_lambda * l2_reg
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_edge_pred, val_purpose_pred, val_time_pred, val_joint_pred, val_person_idx, val_location_idx = model(
                val_graph.x_dict,
                val_graph.edge_index_dict,
                val_graph.edge_attr_dict
            )
            
            val_edge_labels = torch.zeros(len(val_person_idx), device=device)
            val_purpose_labels = torch.zeros(len(val_person_idx), 5, device=device)
            val_time_labels = torch.zeros(len(val_person_idx), device=device)
            val_joint_labels = torch.zeros(len(val_person_idx), device=device)
            
            for i, (p, l) in enumerate(zip(val_person_idx, val_location_idx)):
                mask = (val_pos_edges[:, 0] == p) & (val_pos_edges[:, 1] == l)
                if any(mask):
                    val_edge_labels[i] = 1
                    edge_idx = torch.where(mask)[0][0]
                    attrs = val_edge_attrs[edge_idx]
                    
                    val_purpose_labels[i] = attrs[:5]
                    val_time_labels[i] = attrs[5]
                    val_joint_labels[i] = attrs[6]
            
            val_edge_loss = focal_loss(val_edge_pred.squeeze(), val_edge_labels) * edge_weight
            val_purpose_loss = F.cross_entropy(val_purpose_pred, val_purpose_labels) * purpose_weight
            val_time_loss = F.mse_loss(val_time_pred.squeeze(), val_time_labels) * time_weight
            val_joint_loss = F.binary_cross_entropy(val_joint_pred.squeeze(), val_joint_labels) * joint_weight
            
            val_loss = val_edge_loss + val_purpose_loss + val_time_loss + val_joint_loss
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            logger.info(f'Training Loss: {loss.item():.4f}')
            logger.info(f'Validation Loss: {val_loss.item():.4f}')
            logger.info(f'Edge Loss: {edge_loss.item():.4f}, Purpose Loss: {purpose_loss.item():.4f}')
            logger.info(f'Time Loss: {time_loss.item():.4f}, Joint Loss: {joint_loss.item():.4f}')
            logger.info(f'Edge Prediction Range: [{edge_pred.min().item():.4f}, {edge_pred.max().item():.4f}]')
            logger.info(f'Positive Edge Predictions: {(edge_pred > 0.5).float().mean().item():.4f}')
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses

def evaluate_model(
    model: HeteroGNN,
    test_graph: HeteroData,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained HeteroGNN model
        test_graph: Test graph
        device: Device to evaluate on
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    test_graph = test_graph.to(device)
    
    with torch.no_grad():
        edge_pred, purpose_pred, time_pred, joint_pred, person_idx, location_idx = model(
            test_graph.x_dict,
            test_graph.edge_index_dict,
            test_graph.edge_attr_dict
        )
        
        # Get ground truth edges
        test_edges = test_graph['people', 'visits', 'location'].edge_index
        test_pos_edges = test_edges.t()
        
        # Create labels
        edge_labels = torch.zeros(len(person_idx), device=device)
        purpose_labels = torch.zeros(len(person_idx), 5, device=device)
        time_labels = torch.zeros(len(person_idx), device=device)
        joint_labels = torch.zeros(len(person_idx), device=device)
        
        for i, (p, l) in enumerate(zip(person_idx, location_idx)):
            mask = (test_pos_edges[:, 0] == p) & (test_pos_edges[:, 1] == l)
            if any(mask):
                edge_labels[i] = 1
                edge_idx = torch.where(mask)[0][0]
                attrs = test_graph['people', 'visits', 'location'].edge_attr[edge_idx]
                
                # Purpose (one-hot encoded)
                purpose_labels[i] = attrs[:5]
                # Time (normalized to [0,1])
                time_labels[i] = attrs[5]
                # Joint activity
                joint_labels[i] = attrs[6]
        
        # Calculate metrics
        edge_pred_np = edge_pred.squeeze().cpu().numpy()
        edge_labels_np = edge_labels.cpu().numpy()
        
        edge_auc = roc_auc_score(edge_labels_np, edge_pred_np)
        edge_ap = average_precision_score(edge_labels_np, edge_pred_np)
        
        purpose_accuracy = (torch.argmax(purpose_pred, dim=1) == torch.argmax(purpose_labels, dim=1)).float().mean().item()
        time_mae = torch.abs(time_pred.squeeze() - time_labels).mean().item()
        joint_accuracy = ((joint_pred.squeeze() > 0.5) == (joint_labels > 0.5)).float().mean().item()
        
        return {
            'edge_auc': edge_auc,
            'edge_ap': edge_ap,
            'purpose_accuracy': purpose_accuracy,
            'time_mae': time_mae,
            'joint_accuracy': joint_accuracy
        }

def predict_connections(
    model: HeteroGNN,
    graph: HeteroData,
    device: torch.device,
    threshold: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict people-to-location connections and their attributes for a given graph.
    
    Args:
        model: Trained HeteroGNN model
        graph: Input graph without people-to-location connections
        device: Device to run prediction on
        threshold: Probability threshold for positive predictions
    
    Returns:
        Tuple of (predicted edge indices, edge probabilities, purpose probabilities,
                 time predictions, joint activity probabilities)
    """
    model.eval()
    graph = graph.to(device)
    
    with torch.no_grad():
        # Get predictions for all possible people-location pairs
        edge_pred, purpose_pred, time_pred, joint_pred, person_idx, location_idx = model(
            graph.x_dict,
            graph.edge_index_dict,
            graph.edge_attr_dict
        )
        
        # Log raw prediction statistics
        logger.info(f"Raw prediction statistics:")
        logger.info(f"Edge prediction range: [{edge_pred.min().item():.4f}, {edge_pred.max().item():.4f}]")
        logger.info(f"Mean edge prediction: {edge_pred.mean().item():.4f}")
        
        # Get predictions above threshold
        mask = edge_pred.squeeze() > threshold
        
        # If no predictions above threshold, lower threshold to get top 20% predictions
        if not any(mask):
            logger.info("No predictions above threshold, using top 20% predictions")
            k = max(1, int(len(edge_pred) * 0.2))  # Get top 20% predictions
            _, indices = torch.topk(edge_pred.squeeze(), k)
            mask = torch.zeros_like(edge_pred.squeeze(), dtype=torch.bool)
            mask[indices] = True
        
        # Create edge indices for predicted connections
        pred_edges = torch.stack([
            person_idx[mask],
            location_idx[mask]
        ])
        
        # Get corresponding predictions
        pred_probs = edge_pred[mask]
        pred_purposes = purpose_pred[mask]
        pred_times = time_pred[mask]
        pred_joint = joint_pred[mask]
        
        # Log prediction information
        logger.info(f"Total possible pairs: {len(person_idx)}")
        logger.info(f"Predicted connections: {pred_edges.size(1)}")
        if pred_edges.size(1) > 0:
            logger.info(f"Prediction probability range: [{pred_probs.min().item():.4f}, {pred_probs.max().item():.4f}]")
            logger.info(f"Mean prediction probability: {pred_probs.mean().item():.4f}")
        
        return pred_edges, pred_probs, pred_purposes, pred_times, pred_joint
