#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuroscience-based analysis of LSTM network units
Generate spatial and directional activity maps with quantitative measures
Based on Banino et al. (2018) methodology
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
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
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

class NeuroscienceAnalyzer:
    """Neuroscience-based analysis of LSTM units"""
    
    def __init__(self, spatial_bins=32, directional_bins=20):
        self.spatial_bins = spatial_bins
        self.directional_bins = directional_bins
        
    def create_spatial_ratemap(self, positions, activations, spatial_bins=None):
        """Create spatial ratemap (32x32 grid) for a unit"""
        if spatial_bins is None:
            spatial_bins = self.spatial_bins
            
        # Remove NaN values
        valid_mask = ~(np.isnan(positions).any(axis=1) | np.isnan(activations))
        positions_clean = positions[valid_mask]
        activations_clean = activations[valid_mask]
        
        if len(positions_clean) == 0:
            return np.zeros((spatial_bins, spatial_bins))
        
        # Define spatial bins
        x_min, x_max = positions_clean[:, 0].min(), positions_clean[:, 0].max()
        y_min, y_max = positions_clean[:, 1].min(), positions_clean[:, 1].max()
        
        # Add small padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Create bins
        x_bins = np.linspace(x_min, x_max, spatial_bins + 1)
        y_bins = np.linspace(y_min, y_max, spatial_bins + 1)
        
        # Assign positions to bins
        x_bin_indices = np.digitize(positions_clean[:, 0], x_bins) - 1
        y_bin_indices = np.digitize(positions_clean[:, 1], y_bins) - 1
        
        # Ensure indices are within bounds
        x_bin_indices = np.clip(x_bin_indices, 0, spatial_bins - 1)
        y_bin_indices = np.clip(y_bin_indices, 0, spatial_bins - 1)
        
        # Create ratemap
        ratemap = np.zeros((spatial_bins, spatial_bins))
        occupancy = np.zeros((spatial_bins, spatial_bins))
        
        for i in range(len(positions_clean)):
            x_idx = x_bin_indices[i]
            y_idx = y_bin_indices[i]
            ratemap[y_idx, x_idx] += activations_clean[i]
            occupancy[y_idx, x_idx] += 1
        
        # Calculate mean activity per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            ratemap = np.divide(ratemap, occupancy)
            ratemap[np.isnan(ratemap)] = 0
        
        return ratemap
    
    def create_directional_ratemap(self, head_directions, activations, directional_bins=None):
        """Create directional ratemap (20 bins) for a unit"""
        if directional_bins is None:
            directional_bins = self.directional_bins
            
        # Remove NaN values
        valid_mask = ~(np.isnan(head_directions) | np.isnan(activations))
        head_directions_clean = head_directions[valid_mask]
        activations_clean = activations[valid_mask]
        
        if len(head_directions_clean) == 0:
            return np.zeros(directional_bins)
        
        # Convert to degrees if needed
        if np.max(np.abs(head_directions_clean)) <= np.pi:
            head_directions_deg = np.rad2deg(head_directions_clean)
        else:
            head_directions_deg = head_directions_clean
        
        # Normalize to 0-360 degrees
        head_directions_deg = head_directions_deg % 360
        
        # Create bins
        bin_edges = np.linspace(0, 360, directional_bins + 1)
        bin_indices = np.digitize(head_directions_deg, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, directional_bins - 1)
        
        # Create directional ratemap
        directional_ratemap = np.zeros(directional_bins)
        occupancy = np.zeros(directional_bins)
        
        for i in range(len(head_directions_clean)):
            bin_idx = bin_indices[i]
            directional_ratemap[bin_idx] += activations_clean[i]
            occupancy[bin_idx] += 1
        
        # Calculate mean activity per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            directional_ratemap = np.divide(directional_ratemap, occupancy)
            directional_ratemap[np.isnan(directional_ratemap)] = 0
        
        return directional_ratemap
    
    def calculate_spatial_autocorrelogram(self, ratemap):
        """Calculate spatial autocorrelogram for gridness analysis"""
        # Remove mean
        ratemap_centered = ratemap - np.mean(ratemap)
        
        # Calculate 2D autocorrelation
        autocorr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(ratemap_centered) * 
                                               np.conj(np.fft.fft2(ratemap_centered))))
        autocorr = np.real(autocorr)
        
        # Normalize
        autocorr = autocorr / np.max(autocorr)
        
        return autocorr
    
    def calculate_gridness_score(self, ratemap):
        """Calculate gridness score from spatial autocorrelogram"""
        autocorr = self.calculate_spatial_autocorrelogram(ratemap)
        
        # Get center of autocorrelogram
        center = np.array(autocorr.shape) // 2
        
        # Calculate distances from center
        y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Find peaks at different radii
        max_radius = min(center)
        radii = np.arange(1, max_radius)
        
        correlations = []
        for radius in radii:
            # Get values at this radius
            mask = np.abs(distances - radius) < 0.5
            if np.sum(mask) > 0:
                values = autocorr[mask]
                correlations.append(np.mean(values))
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        
        # Calculate gridness as difference between 60° and 30°/90° correlations
        if len(correlations) < 6:
            return 0
        
        # Find peaks at 60° intervals
        peak_indices = []
        for i in range(6, len(correlations)):
            if (correlations[i] > correlations[i-1] and 
                correlations[i] > correlations[i+1] and
                correlations[i] > 0.3):
                peak_indices.append(i)
        
        if len(peak_indices) < 3:
            return 0
        
        # Calculate gridness
        gridness = 0
        for i in range(len(peak_indices) - 2):
            for j in range(i + 1, len(peak_indices) - 1):
                for k in range(j + 1, len(peak_indices)):
                    # Check if peaks are roughly 60° apart
                    angle1 = np.arctan2(peak_indices[j] - peak_indices[i], 0)
                    angle2 = np.arctan2(peak_indices[k] - peak_indices[j], 0)
                    angle_diff = np.abs(angle1 - angle2)
                    
                    if np.abs(angle_diff - np.pi/3) < np.pi/6:  # 60° ± 30°
                        gridness += correlations[peak_indices[i]] + correlations[peak_indices[j]] + correlations[peak_indices[k]]
        
        return gridness / 3 if gridness > 0 else 0
    
    def calculate_grid_scale(self, ratemap):
        """Calculate grid scale from spatial autocorrelogram"""
        autocorr = self.calculate_spatial_autocorrelogram(ratemap)
        
        # Get center of autocorrelogram
        center = np.array(autocorr.shape) // 2
        
        # Find first peak away from center
        y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Find peak at first ring
        max_radius = min(center)
        for radius in range(1, max_radius):
            mask = np.abs(distances - radius) < 0.5
            if np.sum(mask) > 0:
                values = autocorr[mask]
                if np.max(values) > 0.3:
                    return radius
        
        return 0
    
    def calculate_border_score(self, ratemap, positions):
        """Calculate border score for boundary preference"""
        if len(positions) == 0:
            return 0
        
        # Define environment boundaries
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Calculate distance to nearest border for each bin
        bin_size_x = (x_max - x_min) / ratemap.shape[1]
        bin_size_y = (y_max - y_min) / ratemap.shape[0]
        
        border_distances = np.zeros_like(ratemap)
        for i in range(ratemap.shape[0]):
            for j in range(ratemap.shape[1]):
                # Bin center coordinates
                bin_x = x_min + (j + 0.5) * bin_size_x
                bin_y = y_min + (i + 0.5) * bin_size_y
                
                # Distance to nearest border
                dist_to_borders = [
                    bin_x - x_min,  # left
                    x_max - bin_x,  # right
                    bin_y - y_min,  # bottom
                    y_max - bin_y   # top
                ]
                border_distances[i, j] = min(dist_to_borders)
        
        # Calculate correlation between firing rate and border distance
        valid_mask = ratemap > 0
        if np.sum(valid_mask) < 10:
            return 0
        
        correlation = np.corrcoef(ratemap[valid_mask], border_distances[valid_mask])[0, 1]
        
        # Border score is negative correlation (higher firing near borders)
        return -correlation if not np.isnan(correlation) else 0
    
    def calculate_resultant_vector_length(self, directional_ratemap):
        """Calculate resultant vector length for directional tuning"""
        if len(directional_ratemap) == 0:
            return 0
        
        # Convert to radians
        angles = np.linspace(0, 2*np.pi, len(directional_ratemap), endpoint=False)
        
        # Calculate mean resultant vector
        x_component = np.sum(directional_ratemap * np.cos(angles))
        y_component = np.sum(directional_ratemap * np.sin(angles))
        
        # Resultant vector length
        resultant_length = np.sqrt(x_component**2 + y_component**2) / np.sum(directional_ratemap)
        
        return resultant_length if not np.isnan(resultant_length) else 0
    
    def rayleigh_test(self, directional_ratemap, alpha=0.01):
        """Rayleigh test for directional uniformity"""
        if len(directional_ratemap) == 0:
            return False, 0
        
        # Convert to radians
        angles = np.linspace(0, 2*np.pi, len(directional_ratemap), endpoint=False)
        
        # Calculate test statistic
        n = np.sum(directional_ratemap)
        if n == 0:
            return False, 0
        
        x_component = np.sum(directional_ratemap * np.cos(angles))
        y_component = np.sum(directional_ratemap * np.sin(angles))
        
        R = np.sqrt(x_component**2 + y_component**2)
        z = R**2 / n
        
        # Critical value for alpha = 0.01
        z_critical = 6.63  # Approximate for large n
        
        return z > z_critical, z
    
    def spatial_field_shuffle(self, ratemap, n_shuffles=500):
        """Spatial field shuffle for null distribution"""
        shuffled_ratemaps = []
        
        for _ in range(n_shuffles):
            # Create shuffled version by rotating and flipping
            shuffled = ratemap.copy()
            
            # Random rotation
            if np.random.random() > 0.5:
                shuffled = np.rot90(shuffled, k=np.random.randint(1, 4))
            
            # Random flip
            if np.random.random() > 0.5:
                shuffled = np.fliplr(shuffled)
            if np.random.random() > 0.5:
                shuffled = np.flipud(shuffled)
            
            shuffled_ratemaps.append(shuffled)
        
        return shuffled_ratemaps
    
    def calculate_discreteness_measure(self, scales):
        """Calculate discreteness measure for scale distribution"""
        if len(scales) < 3:
            return 0
        
        # Sort scales
        sorted_scales = np.sort(scales)
        
        # Calculate gaps between consecutive scales
        gaps = np.diff(sorted_scales)
        
        # Calculate discreteness as ratio of largest gap to mean gap
        if np.mean(gaps) == 0:
            return 0
        
        discreteness = np.max(gaps) / np.mean(gaps)
        return discreteness
    
    def gaussian_mixture_clustering(self, scales, max_components=5):
        """Fit Gaussian mixture model to scale distribution"""
        if len(scales) < 3:
            return None, None
        
        scales = np.array(scales).reshape(-1, 1)
        
        # Try different numbers of components
        best_bic = np.inf
        best_model = None
        best_n_components = 1
        
        for n_components in range(1, min(max_components + 1, len(scales))):
            try:
                model = GaussianMixture(n_components=n_components, random_state=42)
                model.fit(scales)
                bic = model.bic(scales)
                
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_n_components = n_components
            except:
                continue
        
        return best_model, best_n_components

def analyze_lstm_units(model, features, positions, head_directions, device=device):
    """Analyze LSTM units using neuroscience methods"""
    model.eval()
    
    # Get LSTM activations
    all_lstm_activations = []
    chunk_size = 1000
    
    for i in range(0, len(features), chunk_size):
        chunk_features = features[i:i+chunk_size]
        test_input = torch.FloatTensor(chunk_features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            lstm_output, _ = model.lstm(test_input)
            lstm_activations = lstm_output.squeeze(0).cpu().numpy()
            all_lstm_activations.append(lstm_activations)
    
    lstm_activations = np.vstack(all_lstm_activations)
    
    # Initialize analyzer
    analyzer = NeuroscienceAnalyzer()
    
    # Analyze each unit
    n_units = lstm_activations.shape[1]
    results = {
        'unit_indices': [],
        'spatial_ratemaps': [],
        'directional_ratemaps': [],
        'gridness_scores': [],
        'grid_scales': [],
        'border_scores': [],
        'resultant_vector_lengths': [],
        'rayleigh_test_results': [],
        'is_grid_like': [],
        'is_border_like': [],
        'is_directionally_modulated': []
    }
    
    print(f"Analyzing {n_units} LSTM units...")
    
    for unit_idx in range(n_units):
        if unit_idx % 10 == 0:
            print(f"Processing unit {unit_idx}/{n_units}")
        
        unit_activations = lstm_activations[:, unit_idx]
        
        # Create spatial and directional ratemaps
        spatial_ratemap = analyzer.create_spatial_ratemap(positions, unit_activations)
        directional_ratemap = analyzer.create_directional_ratemap(head_directions, unit_activations)
        
        # Calculate quantitative measures
        gridness_score = analyzer.calculate_gridness_score(spatial_ratemap)
        grid_scale = analyzer.calculate_grid_scale(spatial_ratemap)
        border_score = analyzer.calculate_border_score(spatial_ratemap, positions)
        resultant_length = analyzer.calculate_resultant_vector_length(directional_ratemap)
        is_directional, rayleigh_stat = analyzer.rayleigh_test(directional_ratemap)
        
        # Classify units
        is_grid_like = gridness_score > 0.37
        is_border_like = border_score > 0.50
        is_directionally_modulated = is_directional
        
        # Store results
        results['unit_indices'].append(unit_idx)
        results['spatial_ratemaps'].append(spatial_ratemap)
        results['directional_ratemaps'].append(directional_ratemap)
        results['gridness_scores'].append(gridness_score)
        results['grid_scales'].append(grid_scale)
        results['border_scores'].append(border_score)
        results['resultant_vector_lengths'].append(resultant_length)
        results['rayleigh_test_results'].append(rayleigh_stat)
        results['is_grid_like'].append(is_grid_like)
        results['is_border_like'].append(is_border_like)
        results['is_directionally_modulated'].append(is_directionally_modulated)
    
    return results, lstm_activations

def generate_neuroscience_analysis_pdf(results, lstm_activations, positions, head_directions, 
                                     save_path="/media/ubuntu/sda/AD_grid/lstm_neuroscience_analysis.pdf"):
    """Generate comprehensive neuroscience analysis PDF"""
    
    with PdfPages(save_path) as pdf:
        # Overview page
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.5, alpha=0.7)
        axes[0, 0].set_title('Mouse Movement Trajectory', fontsize=14)
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Unit classification summary
        n_units = len(results['unit_indices'])
        n_grid_like = sum(results['is_grid_like'])
        n_border_like = sum(results['is_border_like'])
        n_directional = sum(results['is_directionally_modulated'])
        
        categories = ['Grid-like', 'Border-like', 'Directional', 'Other']
        counts = [n_grid_like, n_border_like, n_directional, 
                 n_units - n_grid_like - n_border_like - n_directional]
        
        axes[0, 1].bar(categories, counts, color=['red', 'blue', 'green', 'gray'])
        axes[0, 1].set_title('Unit Classification Summary', fontsize=14)
        axes[0, 1].set_ylabel('Number of Units')
        
        # Gridness score distribution
        axes[0, 2].hist(results['gridness_scores'], bins=30, alpha=0.7, color='red')
        axes[0, 2].axvline(0.37, color='black', linestyle='--', label='Threshold')
        axes[0, 2].set_title('Gridness Score Distribution', fontsize=14)
        axes[0, 2].set_xlabel('Gridness Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # Border score distribution
        axes[1, 0].hist(results['border_scores'], bins=30, alpha=0.7, color='blue')
        axes[1, 0].axvline(0.50, color='black', linestyle='--', label='Threshold')
        axes[1, 0].set_title('Border Score Distribution', fontsize=14)
        axes[1, 0].set_xlabel('Border Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Resultant vector length distribution
        axes[1, 1].hist(results['resultant_vector_lengths'], bins=30, alpha=0.7, color='green')
        axes[1, 1].axvline(0.47, color='black', linestyle='--', label='Threshold')
        axes[1, 1].set_title('Resultant Vector Length Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Resultant Vector Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # Grid scale distribution (for grid-like units only)
        grid_scales = [scale for i, scale in enumerate(results['grid_scales']) 
                      if results['is_grid_like'][i] and scale > 0]
        if grid_scales:
            axes[1, 2].hist(grid_scales, bins=20, alpha=0.7, color='orange')
            axes[1, 2].set_title('Grid Scale Distribution (Grid-like Units)', fontsize=14)
            axes[1, 2].set_xlabel('Grid Scale')
            axes[1, 2].set_ylabel('Frequency')
        else:
            axes[1, 2].text(0.5, 0.5, 'No grid-like units found', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Grid Scale Distribution', fontsize=14)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Individual unit analysis pages
        units_per_page = 16
        n_units = len(results['unit_indices'])
        
        for page_start in range(0, n_units, units_per_page):
            page_end = min(page_start + units_per_page, n_units)
            n_units_this_page = page_end - page_start
            
            # Calculate subplot layout
            n_cols = 4
            n_rows = (n_units_this_page + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, unit_idx in enumerate(range(page_start, page_end)):
                row = i // n_cols
                col = i % n_cols
                
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                # Get unit data
                spatial_ratemap = results['spatial_ratemaps'][unit_idx]
                directional_ratemap = results['directional_ratemaps'][unit_idx]
                gridness = results['gridness_scores'][unit_idx]
                border_score = results['border_scores'][unit_idx]
                resultant_length = results['resultant_vector_lengths'][unit_idx]
                
                # Plot spatial ratemap
                im = ax.imshow(spatial_ratemap, cmap='viridis', aspect='equal')
                ax.set_title(f'Unit {unit_idx}\nGrid: {gridness:.3f}, Border: {border_score:.3f}\nDir: {resultant_length:.3f}', 
                           fontsize=10)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
            
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
        
        # Scale clustering analysis (if grid-like units exist)
        grid_scales = [scale for i, scale in enumerate(results['grid_scales']) 
                      if results['is_grid_like'][i] and scale > 0]
        
        if len(grid_scales) >= 3:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Discreteness measure
            analyzer = NeuroscienceAnalyzer()
            discreteness = analyzer.calculate_discreteness_measure(grid_scales)
            
            axes[0].hist(grid_scales, bins=20, alpha=0.7, color='orange')
            axes[0].set_title(f'Grid Scale Distribution\nDiscreteness: {discreteness:.3f}', fontsize=14)
            axes[0].set_xlabel('Grid Scale')
            axes[0].set_ylabel('Frequency')
            
            # Gaussian mixture clustering
            model, n_components = analyzer.gaussian_mixture_clustering(grid_scales)
            if model is not None:
                # Plot fitted components
                x_range = np.linspace(min(grid_scales), max(grid_scales), 100)
                y_pred = model.predict(np.array(grid_scales).reshape(-1, 1))
                
                for i in range(n_components):
                    component_data = [grid_scales[j] for j in range(len(grid_scales)) if y_pred[j] == i]
                    if component_data:
                        axes[1].hist(component_data, bins=10, alpha=0.7, 
                                   label=f'Component {i+1}')
                
                axes[1].set_title(f'Gaussian Mixture Clustering\n{n_components} Components', fontsize=14)
                axes[1].set_xlabel('Grid Scale')
                axes[1].set_ylabel('Frequency')
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, 'Clustering failed', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Gaussian Mixture Clustering', fontsize=14)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"Neuroscience analysis PDF saved to: {save_path}")

# Main workflow
def main():
    print("Generating neuroscience-based LSTM analysis...")
    
    # Process data
    features, positions = preprocess_trajectory_data(wtydir1_data)
    head_directions = np.deg2rad(wtydir1_data['sheaddir'].values)
    
    print(f"Feature shape: {features.shape}")
    print(f"Position shape: {positions.shape}")
    print(f"Head direction shape: {head_directions.shape}")
    
    # Create network instance
    model = SimplePathIntegrationLSTM(
        input_dim=3,
        hidden_dim=256,
        linear_dim=256,
        n_position_cells=200,
        n_head_direction_cells=72,
        dropout=0.3
    ).to(device)
    
    # Load trained model
    try:
        checkpoint = torch.load("/media/ubuntu/sda/AD_grid/spatial_heatmap_lstm_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model successfully")
    except:
        print("Could not load trained model, using random weights")
    
    # Analyze LSTM units using neuroscience methods
    print("Analyzing LSTM units using neuroscience methods...")
    results, lstm_activations = analyze_lstm_units(
        model, features, positions, head_directions
    )
    
    print(f"Analysis completed for {len(results['unit_indices'])} units")
    print(f"Grid-like units: {sum(results['is_grid_like'])}")
    print(f"Border-like units: {sum(results['is_border_like'])}")
    print(f"Directionally modulated units: {sum(results['is_directionally_modulated'])}")
    
    # Generate neuroscience analysis PDF
    print("Generating neuroscience analysis PDF...")
    generate_neuroscience_analysis_pdf(results, lstm_activations, positions, head_directions)
    
    print("Neuroscience analysis completed!")

if __name__ == "__main__":
    main()
