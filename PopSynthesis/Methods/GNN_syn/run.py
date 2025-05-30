"""Main script to run the GNN-based population synthesis model."""

import torch
from create_mock import create_mock_data
from graph import create_graph, create_prediction_graph
from model import HeteroGNN, train_model, evaluate_model, predict_connections
from visualise_graph import visualize_graph, visualize_predictions
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create mock data
    print("Creating mock data...")
    people, locations, activity_df, location_edges = create_mock_data()
    
    # Split activity data into train/val/test
    print("Splitting data into train/val/test sets...")
    train_activity, temp_activity = train_test_split(activity_df, test_size=0.3, random_state=42)
    val_activity, test_activity = train_test_split(temp_activity, test_size=0.5, random_state=42)
    
    # Create graphs for each split
    print("Creating graphs...")
    train_graph = create_graph(people, locations, location_edges, train_activity)
    val_graph = create_graph(people, locations, location_edges, val_activity)
    test_graph = create_graph(people, locations, location_edges, test_activity)
    
    # Visualize the training graph
    print("Visualizing training graph...")
    visualize_graph(train_graph, title="Training Graph")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    print("Initializing model...")
    model = HeteroGNN(hidden_channels=128, num_layers=3, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training parameters
    num_epochs = 2000
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model=model,
        train_graph=train_graph,
        val_graph=val_graph,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        patience=50,
        min_delta=1e-5
    )
    
    # Plot training results
    print("Plotting training results...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_graph, device)
    print("Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Make predictions on new data
    print("Making predictions...")
    pred_graph = create_prediction_graph(people, locations, location_edges)
    
    # Get predictions
    pred_edges, pred_probs, pred_purposes, pred_times, pred_joint = predict_connections(
        model=model,
        graph=pred_graph,
        device=device,
        threshold=0.3  # Lower threshold to show more connections
    )
    
    # Convert predictions to the format expected by visualization
    predictions = []
    purpose_map = {0: 'work', 1: 'shopping', 2: 'home', 3: 'school', 4: 'leisure'}
    
    if pred_edges.size(1) > 0:  # Only process if we have predictions
        for i in range(len(pred_edges[0])):
            person_id = pred_edges[0][i].item() + 1
            location_id = pred_edges[1][i].item() + 1
            prob = pred_probs[i].item()
            purpose = purpose_map[torch.argmax(pred_purposes[i]).item()]
            time = pred_times[i].item() * 24  # Convert to hours
            joint = "Yes" if pred_joint[i].item() > 0.5 else "No"
            
            predictions.append({
                'person_id': person_id,
                'location_id': location_id,
                'probability': prob,
                'purpose': purpose,
                'time': time,
                'joint': joint
            })
    
    # Get actual connections from test set
    actual_connections = []
    if 'people' in test_graph.edge_index_dict and 'visits' in test_graph.edge_index_dict and 'location' in test_graph.edge_index_dict:
        edges = test_graph['people', 'visits', 'location'].edge_index
        attrs = test_graph['people', 'visits', 'location'].edge_attr
        
        for i in range(edges.size(1)):
            person_id = edges[0][i].item() + 1
            location_id = edges[1][i].item() + 1
            purpose = purpose_map[torch.argmax(attrs[i][:5]).item()]
            time = attrs[i][5].item() * 24
            joint = "Yes" if attrs[i][6].item() > 0.5 else "No"
            
            actual_connections.append({
                'person_id': person_id,
                'location_id': location_id,
                'purpose': purpose,
                'time': time,
                'joint': joint
            })
    
    # Print predictions
    logger.info("\nPredicted Connections:")
    logger.info("=" * 80)
    logger.info(f"{'Person ID':<10} {'Location ID':<12} {'Probability':<12} {'Purpose':<10} {'Time':<8} {'Joint':<6}")
    logger.info("-" * 80)
    
    if predictions:
        for pred in predictions:
            logger.info(f"{pred['person_id']:<10} {pred['location_id']:<12} {pred['probability']:.4f}      {pred['purpose']:<10} {pred['time']:.1f}    {pred['joint']:<6}")
    else:
        logger.info("No predictions made above threshold")
    
    # Print actual connections if available
    if actual_connections:
        logger.info("\nActual Connections:")
        logger.info("=" * 80)
        logger.info(f"{'Person ID':<10} {'Location ID':<12} {'Purpose':<10} {'Time':<8} {'Joint':<6}")
        logger.info("-" * 80)
        
        for conn in actual_connections:
            logger.info(f"{conn['person_id']:<10} {conn['location_id']:<12} {conn['purpose']:<10} {conn['time']:.1f}    {conn['joint']:<6}")
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(
        graph=pred_graph,
        predictions=predictions,
        actual_connections=actual_connections,
        title="Predicted vs Actual Connections"
    )

if __name__ == "__main__":
    main() 