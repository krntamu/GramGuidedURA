#!/usr/bin/env python3
"""
Simple test for Flow Model matrix denoising
"""

import torch
import os
import glob

def find_latest_flow_model():
    """Find the latest Flow Model checkpoint"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None
    
    # Look for Flow Model directories (exclude DM directories)
    flow_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and not item.startswith(('dm_', 'best_models_dm')):
            # Check if it has train_models directory
            train_models_path = os.path.join(item_path, 'train_models')
            if os.path.exists(train_models_path):
                flow_dirs.append(item_path)
    
    if not flow_dirs:
        return None
    
    # Get the most recent directory
    latest_dir = max(flow_dirs, key=os.path.getctime)
    train_models_path = os.path.join(latest_dir, 'train_models')
    
    # Find the latest model file
    model_files = glob.glob(os.path.join(train_models_path, 'model-*.pt'))
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def test_matrix_denoising():
    """Test if Flow Model can correctly perform matrix denoising"""
    print("=" * 50)
    print("FLOW MODEL MATRIX DENOISING TEST")
    print("=" * 50)
    
    try:
        from DMCE import FlowModel, CNN
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Try to find pre-trained Flow Model
        model_path = find_latest_flow_model()
        
        if model_path is None:
            print("ERROR: No pre-trained Flow Model found!")
            print("\nTo train a Flow Model, run:")
            print("  python flow_cnn.py --device cuda")
            print("\nOr for quick testing:")
            print("  python flow_cnn.py --device cuda --quick --epochs 10")
            print("\nCreating untrained model for demonstration...")
            
            # Create untrained model for demonstration
            data_shape = (2, 8, 4)  # 2 channels, 8 RX, 4 TX
            network = CNN(
                data_shape=data_shape,
                n_layers_pre=1,
                n_layers_post=1,
                ch_layers_pre=(2, 4),
                ch_layers_post=(4, 2),
                n_layers_time=1,
                ch_init_time=4,
                kernel_size=(3, 3),
                mode='2D',
                device=device
            )
            
            flow_model = FlowModel(
                model=network,
                data_shape=data_shape,
                complex_data=True,
                loss_type='l2',
                num_timesteps=10,
                sigma_min=0.01,
                sigma_max=2.0,
                rho=7.0,
                device=device
            )
            
            print(f"WARNING: Using untrained model with {flow_model.num_parameters:,} parameters")
            print("         (Results will be poor - please train first!)")
            
        else:
            print(f"Found pre-trained Flow Model: {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract model parameters from checkpoint
            data_shape = checkpoint.get('data_shape', (2, 64, 16))  # Default fallback
            complex_data = checkpoint.get('complex_data', True)
            loss_type = checkpoint.get('loss_type', 'l2')
            num_timesteps = checkpoint.get('num_timesteps', 100)
            sigma_min = checkpoint.get('sigma_min', 0.01)
            sigma_max = checkpoint.get('sigma_max', 50.0)
            rho = checkpoint.get('rho', 7.0)
            
            print(f"  Data shape: {data_shape}")
            print(f"  Complex data: {complex_data}")
            print(f"  Timesteps: {num_timesteps}")
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            
            # Create network with same architecture
            network = CNN(
                data_shape=data_shape,
                n_layers_pre=1,
                n_layers_post=1,
                ch_layers_pre=(2, 4),
                ch_layers_post=(4, 2),
                n_layers_time=1,
                ch_init_time=4,
                kernel_size=(3, 3),
                mode='2D',
                device=device
            )
            
            flow_model = FlowModel(
                model=network,
                data_shape=data_shape,
                complex_data=complex_data,
                loss_type=loss_type,
                num_timesteps=num_timesteps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                device=device
            )
            
            # Load trained weights
            flow_model.model.load_state_dict(checkpoint['model'])
            print("  Loaded trained weights")
        
        # Create test data
        batch_size = 2
        true_matrix = torch.randn(batch_size, *data_shape, device=device)
        
        # Add noise to create noisy observation
        noise_level = 0.3
        noise = noise_level * torch.randn_like(true_matrix)
        noisy_matrix = true_matrix + noise
        
        print(f"\nTest Data:")
        print(f"  True matrix shape: {true_matrix.shape}")
        print(f"  Noisy matrix shape: {noisy_matrix.shape}")
        print(f"  Noise level: {noise_level}")
        
        # Test matrix denoising
        print("\n--- Testing Matrix Denoising ---")
        
        # Method 1: Original denoising
        print("1. Original denoising:")
        snr = 1.0 / (noise_level ** 2)
        x_original = flow_model.generate_estimate(noisy_matrix, snr=snr)
        mse_original = torch.mean((x_original - true_matrix) ** 2).item()
        print(f"  Output shape: {x_original.shape}")
        print(f"  MSE vs true: {mse_original:.6f}")
        
        # Method 2: Conditional denoising
        print("\n2. Conditional denoising:")
        x_conditional = flow_model.generate_conditional_estimate(
            y=noisy_matrix,
            snr=snr,
            n_samples=3,
            n_steps=20
        )
        print(f"  Output shape: {x_conditional.shape}")
        
        # Calculate MSE for conditional estimates
        x_conditional_mean = x_conditional.mean(dim=0)  # Average over samples
        mse_conditional = torch.mean((x_conditional_mean - true_matrix) ** 2).item()
        print(f"  MSE vs true (mean): {mse_conditional:.6f}")
        
        # Compare methods
        print(f"\n--- Comparison ---")
        print(f"Original MSE:     {mse_original:.6f}")
        print(f"Conditional MSE:  {mse_conditional:.6f}")
        
        if mse_conditional < mse_original:
            improvement = (mse_original - mse_conditional) / mse_original * 100
            print(f"Conditional method improves by {improvement:.2f}%")
        else:
            print("Conditional method does not improve")
        
        print("\nMatrix denoising test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_matrix_denoising()
    if not success:
        exit(1) 