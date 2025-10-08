#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LSTM spatial activity heatmap PDF
"""

import sys
import matplotlib.pyplot as plt
import json
from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading data...")
grid_dir_separated = loadmat("/media/ubuntu/sda/AD_grid/Figure_1/grid_dir_separated_into_hexagonal_and_rectangular.mat")
wtydir1 = pd.DataFrame(grid_dir_separated['wtydir1'])
wtydir1_data = loadmat(f"/media/ubuntu/sda/AD_grid/cleaned_mat/{str(wtydir1[2][0][0])}_cleaned.mat")
wtydir1_data = wtydir1_data['cleaned_data']
wtydir1_data = pd.DataFrame(wtydir1_data, columns=['x', 'y', 'vel', 'sx', 'sy', 'svel', 'headdir', 'sheaddir', 'ind', 'ts'])

print(f"Data shape: {wtydir1_data.shape}")

# Data preprocessing: extract motion trajectory features
def preprocess_trajectory_data(data):
    """Preprocess trajectory data, extract velocity, angular velocity features"""
    # Use smoothed data: sx, sy, svel, sheaddir
    linear_velocity = data['svel'].values
    
    # Calculate angular velocity (head direction change rate)
    headdir_rad = np.deg2rad(data['sheaddir'].values)
    angular_velocity = np.diff(headdir_rad, prepend=headdir_rad[0])
    
    # Handle angular velocity jumps (from -π to π or vice versa)
    angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
    
    # Build input features: [linear_velocity, sin(angular_velocity), cos(angular_velocity)]
    features = np.column_stack([
        linear_velocity,
        np.sin(angular_velocity),
        np.cos(angular_velocity)
    ])
    
    # Position information (use smoothed positions)
    positions = np.column_stack([data['sx'].values, data['sy'].values])
    
    return features, positions

# Simple cell activity generator (without softmax)
class SimpleCellActivityGenerator:
    """Simple cell activity generator without softmax normalization"""
    def __init__(self, n_position_cells=100, n_head_direction_cells=36, 
                 position_std=0.1, head_direction_kappa=2.0):
        self.n_position_cells = n_position_cells
        self.n_head_direction_cells = n_head_direction_cells
        self.position_std = position_std
        self.head_direction_kappa = head_direction_kappa
        
        # Generate position cell receptive field centers
        self.position_centers = self._generate_position_centers()
        
        # Generate head direction cell preferred directions
        self.head_direction_centers = np.linspace(0, 2*np.pi, n_head_direction_cells, endpoint=False)
    
    def _generate_position_centers(self):
        """Generate position cell receptive field centers"""
        x_min, x_max = -50, 50
        y_min, y_max = -50, 50
        
        x_centers = np.random.uniform(x_min, x_max, self.n_position_cells)
        y_centers = np.random.uniform(y_min, y_max, self.n_position_cells)
        
        return np.column_stack([x_centers, y_centers])
    
    def generate_position_activity(self, positions):
        """Generate position cell activity based on positions (without softmax)"""
        activity = np.zeros((len(positions), self.n_position_cells))
        
        for i, pos in enumerate(positions):
            # Calculate distance to all receptive field centers
            distances_squared = np.sum((self.position_centers - pos)**2, axis=1)
            
            # Calculate Gaussian function values: exp(-||x - μ_i||² / (2σ²))
            gaussian_values = np.exp(-distances_squared / (2 * self.position_std**2))
            
            # Use Gaussian values directly, no softmax normalization
            activity[i] = gaussian_values
        
        return activity
    
    def generate_head_direction_activity(self, head_directions):
        """Generate head direction cell activity based on head directions (without softmax)"""
        activity = np.zeros((len(head_directions), self.n_head_direction_cells))
        
        for i, hd in enumerate(head_directions):
            # Calculate angle differences with all preferred directions
            angle_diffs = hd - self.head_direction_centers
            
            # Calculate Von Mises distribution values: exp(κ * cos(φ - μ_j))
            von_mises_values = np.exp(self.head_direction_kappa * np.cos(angle_diffs))
            
            # Use Von Mises values directly, no softmax normalization
            activity[i] = von_mises_values
        
        return activity

# Simple LSTM network
class SimplePathIntegrationLSTM(nn.Module):
    """Simple LSTM path integration network"""
    def __init__(self, input_dim=3, hidden_dim=64, linear_dim=256, 
                 n_position_cells=100, n_head_direction_cells=36, dropout=0.3):
        super(SimplePathIntegrationLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_position_cells = n_position_cells
        self.n_head_direction_cells = n_head_direction_cells
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Linear representation layer
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.position_output = nn.Linear(linear_dim, n_position_cells)
        self.head_direction_output = nn.Linear(linear_dim, n_head_direction_cells)
        
    def forward(self, x):
        """Forward pass"""
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Linear representation layer
        linear_output = self.linear_layer(lstm_output)
        
        # Output layer
        position_outputs = self.position_output(linear_output)
        head_direction_outputs = self.head_direction_output(linear_output)
        
        return position_outputs, head_direction_outputs

# Simple loss function
class SimplePathIntegrationLoss(nn.Module):
    """Simple path integration loss function: MSE loss"""
    def __init__(self, position_weight=1.0, head_direction_weight=1.0):
        super(SimplePathIntegrationLoss, self).__init__()
        self.position_weight = position_weight
        self.head_direction_weight = head_direction_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, position_pred, head_direction_pred, 
                position_target, head_direction_target):
        """Calculate loss"""
        position_loss = self.mse_loss(position_pred, position_target)
        head_direction_loss = self.mse_loss(head_direction_pred, head_direction_target)
        
        total_loss = (self.position_weight * position_loss + 
                     self.head_direction_weight * head_direction_loss)
        
        return total_loss, position_loss, head_direction_loss

# Data preprocessing and sequence generation
def create_sequences(features, position_activity, head_direction_activity, 
                    seq_length=50, batch_size=32):
    """Create training sequences"""
    # Ensure data length consistency
    min_length = min(len(features), len(position_activity), len(head_direction_activity))
    features = features[:min_length]
    position_activity = position_activity[:min_length]
    head_direction_activity = head_direction_activity[:min_length]
    
    # Create sequences
    sequences = []
    for i in range(0, len(features) - seq_length, seq_length // 2):
        seq_features = features[i:i+seq_length]
        seq_position = position_activity[i:i+seq_length]
        seq_head_direction = head_direction_activity[i:i+seq_length]
        
        sequences.append({
            'features': torch.FloatTensor(seq_features),
            'position_target': torch.FloatTensor(seq_position),
            'head_direction_target': torch.FloatTensor(seq_head_direction)
        })
    
    # Create data loader
    dataset = TensorDataset(
        torch.stack([s['features'] for s in sequences]),
        torch.stack([s['position_target'] for s in sequences]),
        torch.stack([s['head_direction_target'] for s in sequences])
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=20, 
                clip_grad_norm=0.5, device=device):
    """Train LSTM path integration model"""
    model.train()
    training_history = {
        'epoch': [],
        'total_loss': [],
        'position_loss': [],
        'head_direction_loss': []
    }
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_position_loss = 0.0
        epoch_head_direction_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, position_target, head_direction_target) in enumerate(train_loader):
            # Move to device
            features = features.to(device)
            position_target = position_target.to(device)
            head_direction_target = head_direction_target.to(device)
            
            # Forward pass
            position_pred, head_direction_pred = model(features)
            
            # Calculate loss
            total_loss, position_loss, head_direction_loss = criterion(
                position_pred, head_direction_pred, 
                position_target, head_direction_target
            )
            
            # Check if loss is NaN
            if torch.isnan(total_loss):
                print(f"Detected NaN loss, skipping this batch")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update parameters
            optimizer.step()
            
            # Record loss
            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_head_direction_loss += head_direction_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {total_loss.item():.4f}')
        
        if num_batches > 0:
            # Calculate average loss
            avg_total_loss = epoch_total_loss / num_batches
            avg_position_loss = epoch_position_loss / num_batches
            avg_head_direction_loss = epoch_head_direction_loss / num_batches
            
            # Record history
            training_history['epoch'].append(epoch + 1)
            training_history['total_loss'].append(avg_total_loss)
            training_history['position_loss'].append(avg_position_loss)
            training_history['head_direction_loss'].append(avg_head_direction_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} completed - '
                  f'Total loss: {avg_total_loss:.4f}, '
                  f'Position loss: {avg_position_loss:.4f}, '
                  f'Head direction loss: {avg_head_direction_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} completed - No valid batches')
        
        print('-' * 60)
    
    return training_history

# Analyze LSTM spatial activity
def analyze_lstm_spatial_activity(model, features, positions, seq_length=100, device=device):
    """Analyze LSTM unit spatial activity patterns"""
    model.eval()
    
    # Use all data, not just first 100 points
    test_seq_features = features
    test_seq_positions = positions
    
    # Convert to tensor
    test_input = torch.FloatTensor(test_seq_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass to LSTM layer
        lstm_output, _ = model.lstm(test_input)
        
        # Get LSTM output (1, seq_len, hidden_dim)
        lstm_activations = lstm_output.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)
        
        return lstm_activations, test_seq_positions

# Generate spatial heatmap
def create_spatial_heatmap(positions, activations, grid_size=50):
    """Create spatial heatmap from positions and activations"""
    # Define grid boundaries
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate activations onto grid
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    heatmap_values = griddata(positions, activations, grid_points, method='linear', fill_value=0)
    heatmap_values = heatmap_values.reshape(grid_size, grid_size)
    
    return X_grid, Y_grid, heatmap_values

# Generate spatial activity PDF
def generate_spatial_activity_pdf(lstm_activations, positions, save_path="/media/ubuntu/sda/AD_grid/lstm_spatial_heatmap.pdf"):
    """Generate LSTM unit spatial activity PDF with heatmaps"""
    
    with PdfPages(save_path) as pdf:
        # Overview page
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Real trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.5, alpha=0.7)
        axes[0, 0].set_title('Mouse Movement Trajectory', fontsize=14)
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # LSTM activation heatmap
        im = axes[0, 1].imshow(lstm_activations.T, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('LSTM Unit Activation Heatmap', fontsize=14)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('LSTM Unit Index')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Activation intensity distribution
        valid_activations = lstm_activations[~np.isnan(lstm_activations)]
        if len(valid_activations) > 0:
            axes[1, 0].hist(valid_activations, bins=50, alpha=0.7, color='blue')
            axes[1, 0].set_title('LSTM Activation Intensity Distribution', fontsize=14)
            axes[1, 0].set_xlabel('Activation Intensity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid activation data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('LSTM Activation Intensity Distribution', fontsize=14)
        
        # Unit activation over time
        if not np.all(np.isnan(lstm_activations)):
            axes[1, 1].plot(lstm_activations[:, 0], label='Unit 0', alpha=0.7)
            axes[1, 1].plot(lstm_activations[:, 1], label='Unit 1', alpha=0.7)
            axes[1, 1].plot(lstm_activations[:, 2], label='Unit 2', alpha=0.7)
            axes[1, 1].set_title('LSTM Unit Activation Over Time', fontsize=14)
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Activation Intensity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid activation data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('LSTM Unit Activation Over Time', fontsize=14)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Each LSTM unit spatial activity pattern as heatmap
        n_units = lstm_activations.shape[1]
        units_per_page = 16
        
        for page_start in range(0, n_units, units_per_page):
            page_end = min(page_start + units_per_page, n_units)
            n_units_this_page = page_end - page_start
            
            # Calculate subplot layout
            n_cols = 4
            n_rows = (n_units_this_page + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, unit_idx in enumerate(range(page_start, page_end)):
                row = i // n_cols
                col = i % n_cols
                
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                # Create spatial heatmap
                unit_activations = lstm_activations[:, unit_idx]
                if not np.all(np.isnan(unit_activations)):
                    X_grid, Y_grid, heatmap_values = create_spatial_heatmap(positions, unit_activations)
                    
                    im = ax.contourf(X_grid, Y_grid, heatmap_values, levels=20, cmap='viridis')
                    ax.set_title(f'LSTM Unit {unit_idx}', fontsize=10)
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'LSTM Unit {unit_idx}', fontsize=10)
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
            
            # Hide extra subplots
            for i in range(n_units_this_page, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows == 1:
                    axes[col].set_visible(False)
                else:
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"LSTM spatial activity PDF saved to: {save_path}")

# Main training workflow
def main():
    print("Starting stable LSTM path integration network training...")
    
    # Process data
    features, positions = preprocess_trajectory_data(wtydir1_data)
    print(f"Feature shape: {features.shape}")
    print(f"Position shape: {positions.shape}")
    
    # Create simple cell activity generator
    cell_generator = SimpleCellActivityGenerator(
        n_position_cells=100, 
        n_head_direction_cells=36,
        position_std=0.1,
        head_direction_kappa=2.0
    )
    
    # Generate position cell activity
    position_activity = cell_generator.generate_position_activity(positions)
    
    # Generate head direction cell activity
    headdir_rad = np.deg2rad(wtydir1_data['sheaddir'].values)
    head_direction_activity = cell_generator.generate_head_direction_activity(headdir_rad)
    
    print(f"Position cell activity shape: {position_activity.shape}")
    print(f"Head direction cell activity shape: {head_direction_activity.shape}")
    
    # Create simple network instance
    model = SimplePathIntegrationLSTM(
        input_dim=3,
        hidden_dim=64,
        linear_dim=256,
        n_position_cells=100,
        n_head_direction_cells=36,
        dropout=0.3
    ).to(device)
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = SimplePathIntegrationLoss(position_weight=1.0, head_direction_weight=1.0)
    
    # Create training data
    train_loader = create_sequences(features, position_activity, head_direction_activity, 
                                   seq_length=50, batch_size=32)
    
    print(f"Training batch count: {len(train_loader)}")
    
    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=15,
        clip_grad_norm=0.5,
        device=device
    )
    
    # Analyze LSTM unit spatial activity
    print("Analyzing LSTM unit spatial activity...")
    lstm_activations, test_positions = analyze_lstm_spatial_activity(
        model, features, positions, seq_length=100
    )
    
    # Generate spatial activity PDF
    print("Generating LSTM spatial activity PDF...")
    generate_spatial_activity_pdf(lstm_activations, test_positions)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 3,
            'hidden_dim': 64,
            'linear_dim': 256,
            'n_position_cells': 100,
            'n_head_direction_cells': 36,
            'dropout': 0.3
        }
    }, "/media/ubuntu/sda/AD_grid/spatial_heatmap_lstm_model.pth")
    
    print("Training completed!")
    if training_history['total_loss']:
        print(f"Final loss: {training_history['total_loss'][-1]:.4f}")
    else:
        print("NaN occurred during training, but model saved")
    print("Model and PDF saved")

if __name__ == "__main__":
    main()
