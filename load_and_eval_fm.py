"""
Load and evaluate script for the Flow Model (FMCE).
"""

import os
import argparse
import datetime
import csv
import glob
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import DMCE
import modules.utils as ut
from DMCE.utils import cmplx2real

CUDA_DEFAULT_ID = 0


def find_latest_flow_model():
    """Find the latest Flow Model checkpoint"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None

    flow_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and not item.startswith(('dm_', 'best_models_dm')):
            train_models_path = os.path.join(item_path, 'train_models')
            if os.path.exists(train_models_path):
                flow_dirs.append(item_path)

    if not flow_dirs:
        return None

    latest_dir = max(flow_dirs, key=os.path.getctime)
    train_models_path = os.path.join(latest_dir, 'train_models')
    model_files = glob.glob(os.path.join(train_models_path, 'model-*.pt'))
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model


def get_adaptive_steps(snr_db):
    """
    Get adaptive number of steps based on SNR.
    Based on DM results: [68, 52, 36, 23, 13, 7, 4, 2, 1, 1, 1, 1]
    Compressed to:      [20, 15, 11,  7,  4, 2, 2, 1, 1, 1, 1]
    """
    # DM steps for SNRs: [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    dm_steps = [68, 52, 36, 23, 13, 7, 4, 2, 1, 1, 1] 
    dm_snrs = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]

    # 这里暂时统一测试 1 步；如需用压缩版步数，把下行改成 compressed_steps = [20,15,11,7,4,2,2,1,1,1,1]
    compressed_steps = [68, 52, 36, 23, 13, 7, 4, 2, 1, 1, 1] 

    # Find the closest SNR index
    snr_idx = min(range(len(dm_snrs)), key=lambda i: abs(dm_snrs[i] - snr_db))
    return compressed_steps[snr_idx]


def main():
    parser = argparse.ArgumentParser(description='Load and evaluate Flow Model for Channel Estimation')
    parser.add_argument('--device', '-d', default='cuda', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--model_path', type=str, help='Path to specific model checkpoint (optional)')
    parser.add_argument('--fixed_steps', action='store_true', help='Use fixed steps instead of adaptive')
    parser.add_argument('--num_fixed_steps', type=int, default=100, help='Number of fixed steps (if using fixed steps)')
    args = parser.parse_args()

    # get the used device
    device = args.device
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.set_device(CUDA_DEFAULT_ID)
            device = f'cuda:{CUDA_DEFAULT_ID}'
            print(f"CUDA available: {torch.cuda.get_device_name(CUDA_DEFAULT_ID)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(CUDA_DEFAULT_ID).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    elif device == 'cpu':
        print("Using CPU for evaluation")
    else:
        print(f"WARNING: Unknown device '{device}', falling back to CPU")
        device = 'cpu'

    print("=" * 60)
    print("FLOW MODEL LOAD AND EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Fixed steps: {args.fixed_steps}")
    if args.fixed_steps:
        print(f"Number of fixed steps: {args.num_fixed_steps}")
    else:
        print("Using adaptive steps based on SNR")

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')

    # Data parameters
    n_dim = 64   # RX antennas
    n_dim2 = 16  # TX antennas
    num_train_samples = 100_000
    num_val_samples = 10_000
    num_test_samples = 10_000
    return_all_timesteps = False
    fft_pre = True

    # set data params
    ch_type = '3gpp'
    n_path = 3
    mode = '2D' if n_dim2 > 1 else '1D'

    print(f"\nLoading test dataset: {ch_type}, {num_test_samples} samples")
    try:
        _, _, data_test = ut.load_or_create_data(
            ch_type=ch_type,
            n_path=n_path,
            n_antennas_rx=n_dim,
            n_antennas_tx=n_dim2,
            n_train_ch=num_train_samples,
            n_val_ch=num_val_samples,
            n_test_ch=num_test_samples,
            return_toep=False
        )
        print("Test dataset loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return

    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
        data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
        data_test = cmplx2real(data_test, dim=1, new_dim=False).float()

    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # Find and load the Flow Model
    if args.model_path:
        model_path = args.model_path
        print(f"Using specified model: {model_path}")
    else:
        # Try to find the specific model path you mentioned
        specific_path = "results/2025-11-10-13h16m47s/train_models"
        if os.path.exists(specific_path):
            model_files = glob.glob(os.path.join(specific_path, 'model-*.pt'))
            if model_files:
                model_path = max(model_files, key=os.path.getctime)
                print(f"Found model in specified path: {model_path}")
            else:
                print(f"ERROR: No model files found in {specific_path}")
                return
        else:
            model_path = find_latest_flow_model()
            if model_path is None:
                print("ERROR: No pre-trained Flow Model found!")
                print("Please train a Flow Model first using: python flow_cnn.py")
                return
            print(f"Found latest Flow Model: {model_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("Model checkpoint loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return

    # Extract parameters from checkpoint
    sim_params = checkpoint.get('sim_params', {})
    if not sim_params:
        print("WARNING: No sim_params found in checkpoint, using defaults from flow_cnn.py")

        # Flow model specific parameters (EDM framework)
        sigma_min = 0.01
        sigma_max = 50.0
        rho = 7.0
        noise_std = 1.0
        sampling_eps = 0.002
        num_timesteps = 100
        loss_type = 'l2'

        # CNN architecture parameters (from flow_cnn.py)
        kernel_size = (3, 3)
        n_layers_pre = 2
        max_filter = 64
        ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre + 1, dtype=int)
        ch_layers_pre[0] = 2
        ch_layers_pre = tuple(int(x) for x in ch_layers_pre)

        n_layers_post = 3
        ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post + 1, dtype=int)
        ch_layers_post[0] = 2
        ch_layers_post = ch_layers_post[::-1]
        ch_layers_post = tuple(int(x) for x in ch_layers_post)

        n_layers_time = 1
        ch_init_time = 16
        batch_norm = False
        downsamp_fac = 1

        # Set data shape
        data_shape = tuple(data_test.shape[1:])

        # Create parameter dictionaries
        flow_model_dict = {
            'data_shape': data_shape,
            'complex_data': True,
            'loss_type': loss_type,
            'num_timesteps': num_timesteps,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'rho': rho,
            'noise_std': noise_std,
            'sampling_eps': sampling_eps,
            'clipping': False,
            'device': device
        }
        cnn_dict = {
            'data_shape': data_shape,
            'n_layers_pre': n_layers_pre,
            'n_layers_post': n_layers_post,
            'ch_layers_pre': ch_layers_pre,
            'ch_layers_post': ch_layers_post,
            'n_layers_time': n_layers_time,
            'ch_init_time': ch_init_time,
            'kernel_size': kernel_size,
            'mode': mode,
            'batch_norm': batch_norm,
            'downsamp_fac': downsamp_fac,
            'device': device,
        }
    else:
        # Get model parameters from sim_params
        flow_model_dict = sim_params.get('flow_model_dict', {})
        cnn_dict = sim_params.get('cnn_dict', {})

        # Set device & data shape
        data_shape = tuple(data_test.shape[1:])
        flow_model_dict['device'] = device
        cnn_dict['device'] = device
        flow_model_dict['data_shape'] = data_shape
        cnn_dict['data_shape'] = data_shape

    print(f"Data shape: {data_shape}")
    print(f"Test data: {data_test.shape}")

    # Create models using the exact same parameters as training
    print("\nCreating models...")
    try:
        # Print the actual parameters from checkpoint to debug
        print("Checkpoint keys:", list(checkpoint.keys()))
        if 'sim_params' in checkpoint:
            print("Available sim_params keys:", list(checkpoint['sim_params'].keys()))

        # Use the exact same parameters from training
        cnn = DMCE.CNN(**cnn_dict)
        flow_model = DMCE.FlowModel(cnn, **flow_model_dict)

        # Print model structure to debug
        print("Model state dict keys:", list(flow_model.state_dict().keys()))
        print("Checkpoint model keys:", list(checkpoint['model'].keys()))

        # Handle state dict compatibility issues
        checkpoint_state = checkpoint['model'].copy()

        # NOTE: 如有 key 名不匹配，可在此做兼容处理；当前保留示例逻辑
        if 'sigma_deriv' in checkpoint_state and 'sigma_dt' not in checkpoint_state:
            print("Converting 'sigma_deriv' to 'sigma_dt' for compatibility...")
            checkpoint_state['sigma_dt'] = checkpoint_state.pop('sigma_deriv')

        # Remove keys that don't exist in current model
        current_keys = set(flow_model.state_dict().keys())
        checkpoint_keys = set(checkpoint_state.keys())
        unexpected_keys = checkpoint_keys - current_keys
        if unexpected_keys:
            print(f"Removing unexpected keys: {unexpected_keys}")
            for key in unexpected_keys:
                checkpoint_state.pop(key)

        # Check for missing keys
        missing_keys = current_keys - checkpoint_keys
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
            print("This may cause issues if these are important model parameters.")

        # Load the trained weights
        flow_model.load_state_dict(checkpoint_state, strict=False)
        print("Models created and loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to create models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print model info
    num_timesteps = flow_model_dict.get('num_timesteps', 100)
    print(f"Flow Model timesteps: {num_timesteps}")
    print(f"Number of parameters: {flow_model.num_parameters:,}")

    # Create custom tester with adaptive steps
    class AdaptiveFlowTester:
        def __init__(self, model, data, data_shape, batch_size=512, use_fixed_steps=False, num_fixed_steps=100):
            self.model = model
            self.data = data
            self.data_shape = data_shape
            self.batch_size = batch_size
            self.use_fixed_steps = use_fixed_steps
            self.num_fixed_steps = num_fixed_steps
            self.device = model.device

            # Prepare data loader
            self.num_samples = data.shape[0]
            self.dataloader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=False, pin_memory=True
            )

        def test(self):
            """Test with adaptive or fixed steps"""
            snr_db_range = torch.arange(-10, 45, 5, dtype=torch.float32, device=self.device)
            snr_range = 10 ** (snr_db_range / 10)

            nmse_list = []
            steps_list = []
            timings_sec = []
            tps_ms_list = []

            print(f"\nTesting with {'fixed' if self.use_fixed_steps else 'adaptive'} steps...")
            with torch.no_grad():
                for snr, snr_db in zip(snr_range, snr_db_range):
                    # Determine number of steps
                    if self.use_fixed_steps:
                        n_steps = self.num_fixed_steps
                    else:
                        n_steps = get_adaptive_steps(snr_db.item())

                    print(f"SNR {snr_db.item():2.0f} dB: {n_steps} steps")

                    # COUNT TIME
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    # Test each SNR value
                    x_hat = []
                    for data_batch in self.dataloader:
                        data_batch = data_batch.to(device=self.device)

                        # Add noise to test data
                        y = DMCE.functional.awgn(data_batch, snr, multiplier=self.model.noise_multiplier)

                        # Calculate channel estimate with custom steps
                        x_est = self._generate_estimate_with_steps(y, snr, n_steps)
                        x_hat.append(x_est)

                    x_hat = torch.cat(x_hat, dim=0).cpu()

                    # COUNT TIME END
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    timings_sec.append(dt)
                    tps_ms_list.append(dt * 1000.0 / self.num_samples)

                    # Calculate NMSE
                    if len(self.data.shape) == 4:
                        # Reshape to 2D for NMSE calculation
                        x_hat_reshaped = x_hat.view(x_hat.shape[0], -1)
                        data_reshaped = self.data.view(self.data.shape[0], -1)
                        nmse = DMCE.functional.nmse_torch(
                            data_reshaped, x_hat_reshaped, norm_per_sample=False
                        )
                    else:
                        nmse = DMCE.functional.nmse_torch(
                            torch.squeeze(self.data), torch.squeeze(x_hat), norm_per_sample=False
                        )

                    nmse_list.append(nmse.item() if hasattr(nmse, 'item') else nmse)
                    steps_list.append(n_steps)

            return {
                'SNRs': snr_db_range.tolist(),
                'NMSEs_total_power': nmse_list,
                'Steps': steps_list,
                'Timings_sec': timings_sec,
                'Time_per_sample_ms': tps_ms_list,
            }

        def _generate_estimate_with_steps(self, y, snr, n_steps):
            """Generate estimate with custom number of steps using Flow Model's logic"""

            # If n_steps is 0, return the noisy input directly (no denoising)
            if n_steps == 0:
                return y

            # 用与你先前实现一致的方式：把噪声水平直接作为起点传入
            sigma_N = 1.0 / math.sqrt(float(snr))  # == noise_level

            # NOTE: 你的 FlowModel.reverse_sample_loop 接口是 (x_t, t_start, t_end=None, ..., num_steps=None)
            # 这里把 n_steps 传给 num_steps（原代码写成 n_steps 会报错）
            x_hat = self.model.reverse_sample_loop(
                y, t_start=sigma_N, t_end=0.0, num_steps=n_steps, return_all_timesteps=False
            )
            return x_hat

    # Create tester with adaptive steps (default)
    tester = AdaptiveFlowTester(
        flow_model,
        data_test,
        data_shape,
        batch_size=512,
        use_fixed_steps=args.fixed_steps,
        num_fixed_steps=args.num_fixed_steps
    )

    # Run test using custom AdaptiveFlowTester
    print("\nStarting evaluation...")
    try:
        test_dict = tester.test()
        print("Evaluation completed successfully")
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create results directory
    os.makedirs('./results/fm_est/', exist_ok=True)

    # Prepare results table (using custom tester output format)
    snrs = test_dict['SNRs'].copy()
    nmses = test_dict['NMSEs_total_power'].copy()
    steps = test_dict['Steps'].copy()
    times = test_dict['Timings_sec'].copy()
    tps = test_dict['Time_per_sample_ms'].copy()

    table = []
    table.append(snrs); table[-1].insert(0, 'SNR')
    table.append(nmses); table[-1].insert(0, 'nmse_fm')
    table.append(steps); table[-1].insert(0, 'steps')
    table.append(times); table[-1].insert(0, 'time_s')
    table.append(tps); table[-1].insert(0, 'time_per_sample_ms')

    rows = [list(row) for row in zip(*table)]

    print(f"\nTest Results:")
    for row in rows:
        print(f" {row}")

    # Save detailed results
    step_type = f'fixed_{args.num_fixed_steps}' if args.fixed_steps else 'adaptive'
    file_name = (
        f'./results/fm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_'
        f'T={num_timesteps}_{step_type}_steps_with_time.csv'
    )
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(rows)

    # Save summary results
    mse_list = []
    mse_list.append(test_dict['SNRs'].copy());             mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict['NMSEs_total_power'].copy()); mse_list[-1].insert(0, 'nmse_fm')
    mse_list.append(test_dict['Steps'].copy());             mse_list[-1].insert(0, 'steps')
    mse_list = [list(i) for i in zip(*mse_list)]

    file_name = (
        f'./results/fm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_'
        f'T={num_timesteps}_{step_type}_steps.csv'
    )
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    # Plot results
    file_name = (
        f'./results/fm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_'
        f'T={num_timesteps}_{step_type}_steps.png'
    )

    plt.figure(figsize=(12, 5))

    # Plot NMSE
    plt.subplot(1, 2, 1)
    plt.semilogy(snrs, nmses, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE')
    plt.title(f'Flow Model NMSE vs SNR ({step_type} steps)')
    plt.grid(True, alpha=0.3)

    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(snrs, steps, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Number of Steps')
    plt.title(f'Steps vs SNR ({step_type})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Results plot saved: {file_name}")

    print("\n" + "=" * 60)
    print("FLOW MODEL EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Results saved to: ./results/fm_est/")
    print(f"Step type: {step_type}")

    # Ensure all times are numeric before summing
    total_time = sum(float(t) for t in times)
    print(f"Total test time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
