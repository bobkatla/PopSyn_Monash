import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CategoricalWGAN:
    def __init__(self, categorical_columns, category_dimensions, latent_dim=100, 
                 hidden_dim=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the WGAN model for categorical data.
        
        Args:
            categorical_columns: List of column names for categorical features
            category_dimensions: List of integers representing the number of categories for each column
            latent_dim: Dimension of the noise vector
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on
        """
        self.categorical_columns = categorical_columns
        self.category_dimensions = category_dimensions
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Calculate total output dimension (sum of one-hot encoded dimensions)
        self.output_dim = sum(category_dimensions)
        
        # Generator and Critic
        self.G = Generator(latent_dim, hidden_dim, self.output_dim, category_dimensions).to(device)
        self.D = Critic(self.output_dim, hidden_dim).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.0, 0.9))
        
        # For storing training history
        self.g_losses = []
        self.d_losses = []
    
    def preprocess_data(self, data, ensure_negative_income=True):
        """
        Preprocess the data by one-hot encoding categorical features.
        
        Args:
            data: Pandas DataFrame containing categorical data
            ensure_negative_income: Whether to ensure "Negative income" is included in hhinc categories
        
        Returns:
            Preprocessed data as tensors, encoders for each column
        """
        encoders = {}
        encoded_data = []
        start_idx = 0
        self.column_indices = {}
        
        for i, col in enumerate(self.categorical_columns):
            # Get unique categories for this column
            categories = data[col].unique().tolist()
            
            # Special handling for hhinc to include "Negative income"
            if col == "hhinc" and ensure_negative_income and "Negative income" not in categories:
                categories.append("Negative income")
            
            # Create and fit OneHotEncoder
            # Ensure "Negative income" is always in the categories list
            if col == "hhinc" and "Negative income" not in categories:
                categories.append("Negative income")

            encoder = OneHotEncoder(categories=[categories], sparse_output=False, handle_unknown='ignore')  # ✅ Allow unseen categories

            encoder.fit(data[[col]])
            
            # Transform the data
            encoded_col = encoder.transform(data[[col]])
            encoded_data.append(encoded_col)
            
            # Store encoder and indices for later use
            encoders[col] = encoder
            end_idx = start_idx + len(categories)
            self.column_indices[col] = (start_idx, end_idx)
            start_idx = end_idx
        
        # Concatenate all encoded columns
        all_encoded = np.concatenate(encoded_data, axis=1)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(all_encoded).to(self.device)
        
        return tensor_data, encoders
    
    def train(self, dataloader, epochs=100, n_critic=5, lambda_gp=10):
        """
        Train the WGAN model.
        
        Args:
            dataloader: DataLoader for batched training data
            epochs: Number of training epochs
            n_critic: Number of critic updates per generator update
            lambda_gp: Coefficient for gradient penalty
        """
        for epoch in range(epochs):
            d_loss_epoch = 0
            g_loss_epoch = 0
            
            for batch_idx, real_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                real_data = real_data[0]  # ✅ Extract the actual tensor from DataLoader list
                real_data = real_data.to(self.device)  # ✅ Move tensor to the correct device
                batch_size = real_data.size(0)  # ✅ Now `.size(0)` will work

                real_data = real_data.to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                for _ in range(n_critic):
                    self.d_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, self.latent_dim).to(self.device)
                    fake_data = self.G(z)
                    
                    # Compute critic loss with gradient penalty
                    real_validity = self.D(real_data)
                    fake_validity = self.D(fake_data.detach())
                    
                    # Calculate gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(real_data, fake_data, lambda_gp)
                    
                    # Wasserstein loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                    
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    d_loss_epoch += d_loss.item()
                
                # -----------------
                # Train Generator
                # -----------------
                self.g_optimizer.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.G(z)
                
                # Compute generator loss
                fake_validity = self.D(fake_data)
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
            
            # Record losses
            self.d_losses.append(d_loss_epoch / len(dataloader))
            self.g_losses.append(g_loss_epoch / len(dataloader))
            
            print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss_epoch/len(dataloader):.4f}, G Loss: {g_loss_epoch/len(dataloader):.4f}")
            
            # Check if rare categories are being generated
            if (epoch + 1) % 10 == 0:
                self.check_rare_categories()
    
    def compute_gradient_penalty(self, real_data, fake_data, lambda_gp):
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_data: Batch of real data
            fake_data: Batch of generated data
            lambda_gp: Coefficient for gradient penalty
        
        Returns:
            Gradient penalty term
        """
        batch_size = real_data.size(0)
        
        # Interpolate between real and fake data
        alpha = torch.rand(batch_size, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # Calculate gradients of critic wrt interpolates
        d_interpolates = self.D(interpolates)
        fake = torch.ones(batch_size, 1).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def generate_synthetic_data(self, n_samples, encoders):
        """
        Generate synthetic data using the trained generator.
        
        Args:
            n_samples: Number of samples to generate
            encoders: Dictionary of encoders for each column
        
        Returns:
            Pandas DataFrame of generated synthetic data
        """
        self.G.eval()
        
        # Generate noise
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        
        # Generate one-hot vectors
        with torch.no_grad():
            generated_data = self.G(z).cpu().numpy()
        
        # Convert one-hot back to categorical
        synthetic_data = {}
        
        for col in self.categorical_columns:
            start_idx, end_idx = self.column_indices[col]
            col_one_hot = generated_data[:, start_idx:end_idx]
            
            # Convert to categorial by taking argmax for each row
            col_indices = np.argmax(col_one_hot, axis=1).reshape(-1, 1)
            
            # Use encoder to get categorical values
            encoder = encoders[col]
            categories = encoder.categories_[0]
            col_categories = [categories[idx[0]] for idx in col_indices]
            
            synthetic_data[col] = col_categories
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        
        return synthetic_df
    
    def check_rare_categories(self, n_samples=1000):
        """
        Check if rare categories (like "Negative income") are being generated.
        
        Args:
            n_samples: Number of samples to generate for checking
        """
        self.G.eval()
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        
        with torch.no_grad():
            generated_data = self.G(z).cpu().numpy()
        
        # Check if "Negative income" is present in hhinc column if it exists
        if "hhinc" in self.categorical_columns:
            col_idx = self.categorical_columns.index("hhinc")
            start_idx, end_idx = self.column_indices["hhinc"]
            hhinc_one_hot = generated_data[:, start_idx:end_idx]
            
            # Find the index of "Negative income" if it exists
            # This would be the last category if we added it during preprocessing
            neg_income_idx = end_idx - start_idx - 1
            
            # Count occurrences
            neg_income_count = np.sum(np.argmax(hhinc_one_hot, axis=1) == neg_income_idx)
            neg_income_percentage = (neg_income_count / n_samples) * 100
            
            print(f"'Negative income' generated frequency: {neg_income_percentage:.2f}%")
    
    def plot_losses(self):
        """Plot the training losses for generator and critic."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('WGAN Training Losses')
        plt.show()


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, category_dimensions, temperature=0.5):
        """
        Generator network for categorical data.
        
        Args:
            latent_dim: Dimension of the noise vector
            hidden_dim: Dimension of hidden layers
            output_dim: Total dimension of output (sum of all one-hot dimensions)
            category_dimensions: List of integers representing the number of categories for each column
            temperature: Temperature parameter for Gumbel-Softmax
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.category_dimensions = category_dimensions
        self.temperature = temperature
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
        # Store indices for each categorical feature in the output vector
        self.category_indices = []
        start_idx = 0
        for dim in category_dimensions:
            end_idx = start_idx + dim
            self.category_indices.append((start_idx, end_idx))
            start_idx = end_idx
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Random noise vector (batch_size, latent_dim)
        
        Returns:
            Generated one-hot encoded categorical data
        """
        # Fully connected layers
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply Gumbel-Softmax to each categorical feature separately
        output_parts = []
        
        for start_idx, end_idx in self.category_indices:
            logits = x[:, start_idx:end_idx]
            categorical_probs = self.gumbel_softmax(logits, self.temperature)
            output_parts.append(categorical_probs)
        
        # Concatenate all categorical outputs
        output = torch.cat(output_parts, dim=1)
        
        return output
    
    def gumbel_softmax(self, logits, temperature, hard=True):
        """
        Gumbel-Softmax trick for differentiable categorical sampling.
        
        Args:
            logits: Unnormalized log probabilities
            temperature: Temperature parameter
            hard: If True, the output is one-hot, otherwise it's a probability distribution
        
        Returns:
            Sampled categorical variable (one-hot encoded if hard=True)
        """
        # Sample from Gumbel distribution
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        
        # Softmax
        y_soft = gumbels.softmax(dim=1)
        
        if hard:
            # Straight-through estimator
            index = y_soft.max(dim=1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Critic network for WGAN.
        
        Args:
            input_dim: Dimension of input data (sum of all one-hot dimensions)
            hidden_dim: Dimension of hidden layers
        """
        super(Critic, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        """
        Forward pass through the critic.
        
        Args:
            x: Input data (batch_size, input_dim)
        
        Returns:
            Wasserstein estimate (scalar)
        """
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Function to analyze seed data and extract categorical columns and dimensions
def analyze_seed_data(seed_data):
    """
    Analyze seed data to extract categorical columns and dimensions.
    
    Args:
        seed_data: DataFrame containing the seed data
    
    Returns:
        List of categorical columns and list of category dimensions
    """
    # Identify categorical columns
    categorical_columns = seed_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculate category dimensions for each column
    category_dimensions = []
    for col in categorical_columns:
        unique_values = seed_data[col].unique()
        # Add extra dimension for "Negative income" if it's hhinc column and doesn't contain it already
        if col == "hhinc" and "Negative income" not in unique_values:
            category_dimensions.append(len(unique_values) + 1)
        else:
            category_dimensions.append(len(unique_values))
    
    return categorical_columns, category_dimensions


# Main function to train the model using seed data and generate n agents
def train_popsyn(seed_data, batch_size=64, epochs=100, n_critic=5, lambda_gp=10):
    """
    Train WGAN using seed data and generate n synthetic agents.
    
    Args:
        seed_data: DataFrame containing seed data for training
        n_agents: Number of synthetic agents to generate
        batch_size: Batch size for training
        epochs: Number of training epochs
        n_critic: Number of critic updates per generator update
        lambda_gp: Coefficient for gradient penalty
    
    Returns:
        DataFrame containing synthetic population data
    """
    # Analyze seed data to get categorical columns and dimensions
    categorical_columns, category_dimensions = analyze_seed_data(seed_data)
    print(f"Seed data has {len(seed_data)} samples and {len(categorical_columns)} categorical features.")
    
    # Initialize WGAN model
    wgan = CategoricalWGAN(categorical_columns, category_dimensions)
    
    # Preprocess seed data
    tensor_data, encoders = wgan.preprocess_data(seed_data)
    
    # Create DataLoader
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    print("Starting training...")
    wgan.train(dataloader, epochs=epochs, n_critic=n_critic, lambda_gp=lambda_gp)

    return wgan, encoders, categorical_columns
    

def generate_population(wgan, encoders, categorical_columns, n_agents):
    # Generate synthetic data
    print(f"Generating {n_agents} synthetic agents...")
    synthetic_population = wgan.generate_synthetic_data(n_agents, encoders)
    
    # Verify that rare categories are present
    if 'hhinc' in categorical_columns:
        neg_income_count = (synthetic_population['hhinc'] == 'Negative income').sum()
        print(f"'Negative income' appears in {neg_income_count} agents ({neg_income_count/n_agents*100:.2f}%)")
    
    return synthetic_population


# Example usage
if __name__ == "__main__":

    seed_data = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\GAN\data\hh_sample_ipu.csv")
    seed_data = seed_data.drop(columns=["serialno", "sample_geog"])

    marginals = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\GAN\data\hh_marginals_ipu.csv", header=[0,1])
    marginals = marginals.set_index(marginals.columns[0])
    marginals = marginals.drop(columns=marginals.columns[0])
    totals = marginals.sum(axis=1)/5 #n_atts = 5
    
    # Generate synthetic population
    wgan, encoders, categorical_columns = train_popsyn(seed_data, epochs=50)

    store_pop = []
    for zone, tot in zip(totals.index, totals):
        syn_pop = generate_population(wgan, encoders, categorical_columns, int(tot))
        syn_pop["zone_id"] = zone
        store_pop.append(syn_pop)
    synthetic_population = pd.concat(store_pop)
    
    # Display first few rows of synthetic data
    print("\nSample of generated synthetic agents:")
    print(synthetic_population.head())
    
    # Check distribution of categories in synthetic population
    print("\nDistribution of categories in synthetic population:")
    for col in synthetic_population.columns:
        print(f"\n{col}:")
        print(synthetic_population[col].value_counts(normalize=True) * 100)

    synthetic_population.to_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\GAN\output\wgan_synthetic_population.csv")

    # Plot training losses
    wgan.plot_losses()