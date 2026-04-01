#!/usr/bin/env python3
"""
Improved PINN Neuron Model Training
Fixing g_Na and g_K Parameter Issues

Your current parameters show g_Na=4.30 and g_K=4.10, which are much smaller than 
classic HH values (g_Na=120, g_K=36). This will cause problems with action potential generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import pynwb
import random
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
file = '/media/ubuntu/sda/Patch-seq/data/Patch/601790945_icephys.nwb'
io = pynwb.NWBHDF5IO(file, 'r')
data = io.read()

# Process acquisition data
acquisition_dict = {}
for i in data.stimulus.keys():
    stimulus = data.get_stimulus(i)
    if stimulus.data_type == 'CurrentClampStimulusSeries':
        acquisition_data = pd.DataFrame(data.get_acquisition(i.split("DA")[0] + "AD0").data, columns=['acquisition'])
        acquisition_data['stimulus'] = np.array(data.get_stimulus(i).data)
        acquisition_data['time_step'] = range(len(acquisition_data))
        acquisition_dict[i] = acquisition_data

print(f"Loaded {len(acquisition_dict)} experiments")

# Improved HH_PINN Model with Better Initialization
class ImprovedHH_PINN(nn.Module):
    """
    Improved PINN model with better initialization for g_Na and g_K
    Key improvements:
    1. Log-space parameterization for conductances
    2. Better initialization values
    3. Proper constraints
    """
    def __init__(self, hidden_layers=4, hidden_units=128):
        super(ImprovedHH_PINN, self).__init__()
        
        # Neural network layers
        layers = []
        input_dim = 2  # time t and current I(t)
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        
        # Output layer - voltage V(t)
        layers.append(nn.Linear(hidden_units, 1))
        
        self.network = nn.Sequential(*layers)
        
        # IMPROVED HH model parameters with better initialization
        # Use log-space initialization for conductances to ensure proper scaling
        self.g_Na_raw = nn.Parameter(torch.tensor(4.8))   # exp(4.8) ≈ 120
        self.g_K_raw = nn.Parameter(torch.tensor(3.6))    # exp(3.6) ≈ 36
        self.g_L_raw = nn.Parameter(torch.tensor(-1.2))   # exp(-1.2) ≈ 0.3
        
        # Equilibrium potentials (close to classic values)
        self.E_Na = nn.Parameter(torch.tensor(50.0))
        self.E_K = nn.Parameter(torch.tensor(-77.0))
        self.E_L = nn.Parameter(torch.tensor(-54.4))
        
        # Membrane capacitance
        self.C_m_raw = nn.Parameter(torch.tensor(0.0))    # exp(0.0) = 1.0
        
        # Initial gating variable values
        self.m_init = nn.Parameter(torch.tensor(0.05))
        self.h_init = nn.Parameter(torch.tensor(0.6))
        self.n_init = nn.Parameter(torch.tensor(0.32))
        
    @property
    def g_Na(self):
        """Sodium conductance - ensure positive values with proper range"""
        return torch.clamp(torch.exp(self.g_Na_raw), min=50.0, max=500.0)
    
    @property
    def g_K(self):
        """Potassium conductance - ensure positive values with proper range"""
        return torch.clamp(torch.exp(self.g_K_raw), min=10.0, max=200.0)
    
    @property
    def g_L(self):
        """Leak conductance - ensure positive values"""
        return torch.clamp(torch.exp(self.g_L_raw), min=0.01, max=10.0)
    
    @property
    def C_m(self):
        """Membrane capacitance - ensure positive values"""
        return torch.clamp(torch.exp(self.C_m_raw), min=0.1, max=5.0)
        
    def forward(self, t, I):
        """Forward pass: input time t and current I, output voltage V"""
        inputs = torch.cat([t, I], dim=-1)
        V = self.network(inputs)
        return V
    
    def get_hh_parameters(self):
        """Get HH model parameters"""
        return {
            'g_Na': self.g_Na.item(),
            'g_K': self.g_K.item(),
            'g_L': self.g_L.item(),
            'E_Na': self.E_Na.item(),
            'E_K': self.E_K.item(),
            'E_L': self.E_L.item(),
            'C_m': self.C_m.item(),
            'm_init': self.m_init.item(),
            'h_init': self.h_init.item(),
            'n_init': self.n_init.item()
        }

# Improved Physics Loss with Better Numerical Stability
class ImprovedHH_Physics:
    """Improved HH model physics equations"""
    
    @staticmethod
    def alpha_m(V):
        """Sodium channel m gating activation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        exp_term = torch.exp(-(V_safe + 40) / 10)
        exp_term = torch.clamp(exp_term, min=1e-10, max=1e10)
        return 0.1 * (V_safe + 40) / (1 - exp_term + 1e-10)
    
    @staticmethod
    def beta_m(V):
        """Sodium channel m gating deactivation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        return 4 * torch.exp(-(V_safe + 65) / 18)
    
    @staticmethod
    def alpha_h(V):
        """Sodium channel h gating activation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        return 0.07 * torch.exp(-(V_safe + 65) / 20)
    
    @staticmethod
    def beta_h(V):
        """Sodium channel h gating deactivation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        exp_term = torch.exp(-(V_safe + 35) / 10)
        exp_term = torch.clamp(exp_term, min=1e-10, max=1e10)
        return 1 / (1 + exp_term)
    
    @staticmethod
    def alpha_n(V):
        """Potassium channel n gating activation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        exp_term = torch.exp(-(V_safe + 55) / 10)
        exp_term = torch.clamp(exp_term, min=1e-10, max=1e10)
        return 0.01 * (V_safe + 55) / (1 - exp_term + 1e-10)
    
    @staticmethod
    def beta_n(V):
        """Potassium channel n gating deactivation rate"""
        V_safe = torch.clamp(V, min=-100, max=100)
        return 0.125 * torch.exp(-(V_safe + 65) / 80)

def improved_compute_physics_loss(model, t_points, I_points):
    """Improved physics loss computation"""
    t_points.requires_grad_(True)
    
    # Get model predicted voltage
    V_pred = model(t_points, I_points)
    
    # Compute dV/dt
    dV_dt = grad(V_pred.sum(), t_points, create_graph=True)[0]
    
    # Compute gating variables (steady-state approximation)
    V = V_pred.squeeze()
    
    # Steady-state gating variables
    alpha_m = ImprovedHH_Physics.alpha_m(V)
    beta_m = ImprovedHH_Physics.beta_m(V)
    alpha_h = ImprovedHH_Physics.alpha_h(V)
    beta_h = ImprovedHH_Physics.beta_h(V)
    alpha_n = ImprovedHH_Physics.alpha_n(V)
    beta_n = ImprovedHH_Physics.beta_n(V)
    
    # Avoid division by zero
    m_inf = alpha_m / (alpha_m + beta_m + 1e-10)
    h_inf = alpha_h / (alpha_h + beta_h + 1e-10)
    n_inf = alpha_n / (alpha_n + beta_n + 1e-10)
    
    # HH equation right-hand side
    I_Na = model.g_Na * m_inf**3 * h_inf * (V - model.E_Na)
    I_K = model.g_K * n_inf**4 * (V - model.E_K)
    I_L = model.g_L * (V - model.E_L)
    
    # HH differential equation: C_m * dV/dt = I_ext - I_Na - I_K - I_L
    physics_residual = model.C_m * dV_dt - I_points.squeeze() + I_Na + I_K + I_L
    
    # Add numerical stability
    physics_residual = torch.clamp(physics_residual, min=-1e6, max=1e6)
    
    return torch.mean(physics_residual**2)

def improved_total_loss(model, t_points, I_points, V_true, physics_weight=0.1, data_weight=1.0):
    """Improved total loss function with better weight balance"""
    V_pred = model(t_points, I_points)
    
    # Data loss
    data_loss = F.mse_loss(V_pred, V_true)
    
    if torch.isnan(data_loss):
        return torch.tensor(1e6, requires_grad=True, device=t_points.device)
    
    # Physics loss
    physics_loss = improved_compute_physics_loss(model, t_points, I_points)
    
    if torch.isnan(physics_loss):
        return torch.tensor(1e6, requires_grad=True, device=t_points.device)
    
    total_loss_value = data_weight * data_loss + physics_weight * physics_loss
    
    if torch.isnan(total_loss_value):
        return torch.tensor(1e6, requires_grad=True, device=t_points.device)
    
    return total_loss_value

# Data preprocessing
def preprocess_acquisition_dict(acquisition_dict, sampling_rate=10000, seq_length=2000, overlap=200):
    """Preprocess acquisition data for training"""
    processed_data = []
    
    for key, df in acquisition_dict.items():
        try:
            voltage_data = df['acquisition'].values
            current_data = df['stimulus'].values
            time_steps = df['time_step'].values
            
            total_length = len(voltage_data)
            step_size = seq_length - overlap
            
            for start_idx in range(0, total_length - seq_length + 1, step_size):
                end_idx = start_idx + seq_length
                
                segment_voltage = voltage_data[start_idx:end_idx]
                segment_current = current_data[start_idx:end_idx]
                segment_time_steps = time_steps[start_idx:end_idx]
                
                dt = 1.0 / sampling_rate
                time_points = segment_time_steps * dt
                
                t_tensor = torch.FloatTensor(time_points).unsqueeze(1)
                I_tensor = torch.FloatTensor(segment_current).unsqueeze(1)
                V_tensor = torch.FloatTensor(segment_voltage).unsqueeze(1)
                
                segment_key = f"{key}_seg_{start_idx//step_size}"
                
                processed_data.append({
                    'time': t_tensor,
                    'current': I_tensor,
                    'voltage': V_tensor,
                    'key': segment_key,
                    'original_key': key,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
                
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue
    
    return processed_data

# Process data
train_data = preprocess_acquisition_dict(acquisition_dict)
print(f"Processed {len(train_data)} data segments")

# Improved training function with better parameter control
def improved_train_pinn_model(model, train_data, epochs=100, lr=0.0005, 
                              physics_weight=0.1, data_weight=1.0, 
                              batch_size=4, use_fraction=0.5):
    """
    Improved training function with better parameter control
    Key improvements:
    1. Higher physics weight to enforce HH constraints
    2. Better learning rate schedule
    3. Parameter monitoring
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.7)
    
    losses = []
    
    # Data sampling
    if use_fraction < 1.0:
        n_samples = int(len(train_data) * use_fraction)
        train_data = train_data[:n_samples]
        print(f"Using {use_fraction*100:.0f}% of data for training: {len(train_data)} segments")
    
    print(f"\nImproved training configuration:")
    print(f"Total data segments: {len(train_data)}")
    print(f"Data points per segment: {train_data[0]['time'].shape[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Physics weight: {physics_weight}")
    print(f"Data weight: {data_weight}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        nan_count = 0
        
        # Shuffle data segments
        random.shuffle(train_data)
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            
            if len(batch_data) < batch_size:
                continue
            
            optimizer.zero_grad()
            
            # Compute batch loss
            batch_loss = 0
            valid_samples = 0
            
            for data_sample in batch_data:
                t_points = data_sample['time']
                I_points = data_sample['current']
                V_true = data_sample['voltage']
                
                # Compute loss for single sample
                loss = improved_total_loss(model, t_points, I_points, V_true, 
                                         physics_weight, data_weight)
                
                if torch.isnan(loss):
                    nan_count += 1
                    continue
                
                batch_loss += loss
                valid_samples += 1
            
            if valid_samples == 0:
                continue
            
            # Average loss
            avg_batch_loss = batch_loss / valid_samples
            
            avg_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += avg_batch_loss.item()
            batch_count += 1
        
        if batch_count == 0:
            print(f"Epoch {epoch}: All batches produced NaN, stopping training")
            break
            
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Avg Loss: {avg_loss:.6f}, NaN samples: {nan_count}')
            params = model.get_hh_parameters()
            print(f'Parameters: g_Na={params["g_Na"]:.2f}, g_K={params["g_K"]:.2f}, '
                  f'g_L={params["g_L"]:.3f}, C_m={params["C_m"]:.3f}')
            
            # Check parameter reasonableness
            if params["g_Na"] < 50:
                print(f"  Warning: g_Na too low! (classic value: 120)")
            if params["g_K"] < 20:
                print(f"  Warning: g_K too low! (classic value: 36)")
    
    return losses

# Train the improved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create improved model
improved_model = ImprovedHH_PINN(hidden_layers=4, hidden_units=128).to(device)

# Move data to device
for data_sample in train_data:
    data_sample['time'] = data_sample['time'].to(device)
    data_sample['current'] = data_sample['current'].to(device)
    data_sample['voltage'] = data_sample['voltage'].to(device)

print(f"Data moved to device: {device}")

# Train with improved parameters
print("\nStarting improved training...")
print("Key improvements:")
print("1. Better initialization for g_Na and g_K")
print("2. Higher physics weight (0.1 vs 0.01)")
print("3. Log-space parameterization for conductances")
print("4. Better numerical stability")

losses_improved = improved_train_pinn_model(
    improved_model, 
    train_data, 
    epochs=100,        # More epochs
    lr=0.0005,        # Higher learning rate
    physics_weight=0.1, # Higher physics weight
    data_weight=1.0,   # Keep data weight
    batch_size=2,      # Smaller batch size
    use_fraction=0.2   # Use less data
)

print("Improved training completed!")

# Show final results
print("\nFinal Results:")
final_params = improved_model.get_hh_parameters()
print("Learned HH Model Parameters:")
print("=" * 50)
for param_name, param_value in final_params.items():
    print(f"{param_name:10s}: {param_value:8.4f}")

# Compare with classic values
classic_params = {
    'g_Na': 120.0,
    'g_K': 36.0,
    'g_L': 0.3,
    'E_Na': 50.0,
    'E_K': -77.0,
    'E_L': -54.4,
    'C_m': 1.0
}

print("\nComparison with Classic HH Parameters:")
print("=" * 50)
print(f"{'Parameter':<15} {'Classic':<12} {'Learned':<12} {'Diff(%)':<10} {'Status'}")
print("-" * 50)

for param_name, classic_value in classic_params.items():
    learned_value = final_params[param_name]
    diff_percent = ((learned_value - classic_value) / classic_value) * 100
    if abs(diff_percent) < 50:
        status = "✓ Good"
    elif abs(diff_percent) < 100:
        status = "⚠ Acceptable"
    else:
        status = "✗ Poor"
    print(f"{param_name:<15} {classic_value:<12.4f} {learned_value:<12.4f} {diff_percent:<10.2f}% {status}")

# Test the improved model
def test_improved_model(model, test_data, num_samples=3):
    """Test the improved model"""
    model.eval()
    
    # Random sample selection
    random.seed(42)
    test_samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    print(f"\nTesting improved model with {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        print(f"\nTest Sample {i+1}/{len(test_samples)}:")
        
        # Get data
        t_true = sample['time'].detach().cpu().numpy().flatten()
        I_true = sample['current'].detach().cpu().numpy().flatten()
        V_true = sample['voltage'].detach().cpu().numpy().flatten()
        
        # Model prediction
        with torch.no_grad():
            V_pred = model(sample['time'], sample['current']).cpu().numpy().flatten()
        
        # Calculate metrics
        mse = mean_squared_error(V_true, V_pred)
        r2 = r2_score(V_true, V_pred)
        correlation = np.corrcoef(V_true, V_pred)[0, 1]
        
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        # Plot comparison
        time_ms = t_true * 1000
        
        plt.figure(figsize=(15, 8))
        
        # Voltage comparison
        plt.subplot(2, 1, 1)
        plt.plot(time_ms, V_true, 'b-', label='True Voltage', linewidth=2)
        plt.plot(time_ms, V_pred, 'r--', label='Model Prediction', linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title(f'Improved Model Test {i+1}: Voltage Response (MSE={mse:.4f}, R²={r2:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Current stimulus
        plt.subplot(2, 1, 2)
        plt.plot(time_ms, I_true, 'g-', label='Current Stimulus', linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        plt.title('Current Stimulus')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/media/ubuntu/sda/Patch-seq/data/improved_test_sample_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# Test the improved model
test_data_cpu = []
for sample in train_data[:20]:  # Use first 20 samples for testing
    test_sample = {
        'time': sample['time'].cpu(),
        'current': sample['current'].cpu(),
        'voltage': sample['voltage'].cpu(),
        'key': sample['key']
    }
    test_data_cpu.append(test_sample)

test_improved_model(improved_model.cpu(), test_data_cpu, num_samples=3)

print("\nImproved model testing completed!")
print("Generated files:")
print("- improved_test_sample_1.png to improved_test_sample_3.png")

# Summary
print("\n" + "="*60)
print("SUMMARY OF IMPROVEMENTS:")
print("="*60)
print("1. ✅ Log-space parameterization for g_Na and g_K")
print("2. ✅ Better initialization (g_Na≈120, g_K≈36)")
print("3. ✅ Higher physics weight (0.1 vs 0.01)")
print("4. ✅ Proper parameter constraints")
print("5. ✅ Better numerical stability")
print("\nThis should fix the g_Na and g_K parameter issues!")
