"""
Train and test script for the Flow Model Channel Estimation (FMCE).
"""
from DMCE import utils, FlowModel, FlowTrainer, FlowTester, CNN
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DMCE.utils import cmplx2real

CUDA_DEFAULT_ID = 0


def main():
    parser = argparse.ArgumentParser(description='Train Flow Model for Channel Estimation')
    parser.add_argument('--device', '-d', default='cuda', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--epochs', '-e', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--timesteps', '-t', default=100, type=int, help='Number of timesteps')
    parser.add_argument('--samples', '-s', default=100000, type=int, help='Number of training samples')
    parser.add_argument('--quick', action='store_true', help='Quick training mode with reduced parameters')
    parser.add_argument('--min_epochs', default=5, type=int, help='Minimum number of epochs before early stopping')
    parser.add_argument('--patience', default=5, type=int, help='Number of epochs without improvement before early stopping')
    parser.add_argument('--no_test', action='store_true', help='Skip testing after training')

    # get the used device
    args = parser.parse_args()
    device = args.device
    
    # Set device with better CUDA detection
    if device == 'cuda':
        if torch.cuda.is_available():
            # Set CUDA device
            torch.cuda.set_device(CUDA_DEFAULT_ID)
            device = f'cuda:{CUDA_DEFAULT_ID}'
            print(f"CUDA available: {torch.cuda.get_device_name(CUDA_DEFAULT_ID)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(CUDA_DEFAULT_ID).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    elif device == 'cpu':
        print("Using CPU for training")
    else:
        print(f"WARNING: Unknown device '{device}', falling back to CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("FLOW MODEL CHANNEL ESTIMATION TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Training Parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Early Stopping: min_epochs={args.min_epochs}, patience={args.patience}")

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    # Data parameters
    n_dim = 64 # RX antennas
    n_dim2 = 16 # TX antennas
    num_train_samples = args.samples
    num_val_samples = min(10_000, num_train_samples // 10)  # must not exceed size of training set
    num_test_samples = min(10_000, num_train_samples // 10)
    seed = 453451

    # Quick mode adjustments
    if args.quick:
        print("Quick training mode enabled")
        num_train_samples = min(10000, num_train_samples)
        num_val_samples = min(1000, num_val_samples)
        num_test_samples = min(1000, num_test_samples)
        args.epochs = min(30, args.epochs)
        args.batch_size = min(64, args.batch_size)
        args.min_epochs = min(5, args.min_epochs)
        args.patience = min(5, args.patience)

    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = True # learn channel distribution in angular domain through Fourier transform

    # set data params
    ch_type = '3gpp' # {quadriga_LOS, 3gpp}
    n_path = 3
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'
    complex_data = True

    print(f"\nLoading dataset: {ch_type}, {num_train_samples} samples")
    try:
        data_train, data_val, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                         n_antennas_tx=n_dim2, n_train_ch=num_train_samples, n_val_ch=num_val_samples,
                                         n_test_ch=num_test_samples, return_toep=False)
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return

    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_train = np.reshape(data_train, (-1, n_dim, n_dim2), 'F')
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
        data_val = np.reshape(data_val, (-1, n_dim, n_dim2), 'F')
    
    data_train = torch.from_numpy(np.asarray(data_train[:, None, :]))
    data_train = cmplx2real(data_train, dim=1, new_dim=False).float()
    data_val = torch.from_numpy(np.asarray(data_val[:, None, :]))
    data_val = cmplx2real(data_val, dim=1, new_dim=False).float()
    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()
    
    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # set data params
    cwd = os.getcwd()
    bin_dir = path.join(cwd, 'bin')
    data_shape = tuple(data_train.shape[1:])

    print(f"Data shape: {data_shape}")
    print(f"Training data: {data_train.shape}")
    print(f"Validation data: {data_val.shape}")
    print(f"Test data: {data_test.shape}")

    # data parameter dictionary, which is saved in 'sim_params.json'
    data_dict = {
        'bin_dir': str(bin_dir),
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'num_test_samples': num_test_samples,
        'train_dataset': ch_type,
        'test_dataset': ch_type,
        'n_antennas': n_dim,
        'mode': mode,
        'data_shape': data_shape,
        'complex_data': complex_data
    }

    # set Flow model params
    num_timesteps = args.timesteps
    loss_type = 'l2'
    
    # Flow model specific parameters (EDM framework)
    sigma_min = 0.01
    sigma_max = 50.0
    rho = 7.0
    noise_std = 1.0
    sampling_eps = 0.002

    print(f"\nFlow Model Parameters:")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
    print(f"  Rho: {rho}")

    # flow model parameter dictionary, which is saved in 'sim_params.json'
    flow_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
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

    # CNN architecture parameters
    kernel_size = (3, 3)
    n_layers_pre = 2
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 2
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 3
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 2
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    print(f"CNN Architecture:")
    print(f"  Pre layers: {n_layers_pre} ({ch_layers_pre})")
    print(f"  Post layers: {n_layers_post} ({ch_layers_post})")
    print(f"  Time layers: {n_layers_time}")

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

    # set FlowTrainer params
    batch_size = args.batch_size
    lr_init = args.lr
    lr_step_multiplier = 1.0
    epochs_until_lr_step = 50  # Reduced from 150
    num_epochs = args.epochs
    val_every_n_batches = 1000  # Reduced from 2000 for more frequent validation
    # Early stopping: stop if loss doesn't improve for patience epochs
    # Default: min 5 epochs, patience 5 (stop if no improvement for 5 epochs)
    num_min_epochs = args.min_epochs
    num_epochs_no_improve = args.patience
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)

    print(f"Training Parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr_init}")
    print(f"  Save mode: {save_mode}")
    print(f"  Min epochs: {num_min_epochs}")
    print(f"  Patience: {num_epochs_no_improve}")

    # FlowTrainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        'fft_pre': fft_pre,
    }

    # set FlowTester params
    batch_size_test = 512
    criteria = ['nmse']

    # FlowTester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': batch_size_test,
        'criteria': criteria,
        'complex_data': complex_data,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
    }

    # create result directory
    os.makedirs(dir_result, exist_ok=True)
    print(f"Results directory: {dir_result}")

    # instantiate CNN, FlowModel, FlowTrainer and FlowTester
    print("\nCreating models...")
    try:
        cnn = CNN(**cnn_dict)
        flow_model = FlowModel(cnn, **flow_model_dict)
        trainer = FlowTrainer(flow_model, data_train, data_val, **trainer_dict)
        tester = FlowTester(flow_model, data_test, **tester_dict)
        print("Models created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print number of trainable parameters
    print(f'Number of trainable parameters: {flow_model.num_parameters:,}')

    # other parameters dictionary, which is saved in 'sim_params.json'
    misc_dict = {'num_parameters': flow_model.num_parameters}

    # save the simulation parameters as a JSON file
    sim_dict = {
        'data_dict': data_dict,
        'flow_model_dict': flow_model_dict,
        'cnn_dict': cnn_dict,
        'trainer_dict': trainer_dict,
        'tester_dict': tester_dict,
        'misc_dict': misc_dict
    }

    utils.save_params(dir_result=dir_result, filename='sim_params', params=sim_dict)
    print("Simulation parameters saved")

    # run training routine
    print("\nStarting training...")
    try:
        train_dict = trainer.train()
        print("Training completed successfully")
        utils.save_params(dir_result=dir_result, filename='train_results', params=train_dict)
        print("Training results saved")
        
        # Ensure final model is saved
        print("Saving final model...")
        trainer.save_model(val_loss=train_dict['val_losses'][-1] if train_dict['val_losses'] else 0.0)
        print("Final model saved")
        
        # Verify that model file was created
        train_models_dir = os.path.join(dir_result, 'train_models')
        if os.path.exists(train_models_dir):
            model_files = [f for f in os.listdir(train_models_dir) if f.endswith('.pt')]
            if model_files:
                print(f"Model files found: {model_files}")
                for model_file in model_files:
                    file_path = os.path.join(train_models_dir, model_file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  {model_file}: {file_size:.1f} KB")
            else:
                print("WARNING: No .pt files found in train_models directory")
        else:
            print("WARNING: train_models directory not found")
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save the model even if training failed
        try:
            print("Attempting to save model despite training failure...")
            trainer.save_model(val_loss=float('inf'))  # Use high loss to indicate failure
            print("Model saved despite training failure")
        except Exception as save_error:
            print(f"ERROR: Failed to save model: {save_error}")
        
        return

    # Save training summary
    params = dict()
    params['dim'] = n_dim
    params['dim2'] = n_dim2
    params['data_train'] = num_train_samples
    params['data_test'] = num_test_samples
    params['data_val'] = num_val_samples
    params['epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['lr_start'] = lr_init
    params['lr_step_mult'] = lr_step_multiplier
    params['epochs_until_lr_step'] = epochs_until_lr_step
    params['timesteps'] = num_timesteps
    params['sigma_min'] = sigma_min
    params['sigma_max'] = sigma_max
    params['rho'] = rho
    params['noise_std'] = noise_std
    params['sampling_eps'] = sampling_eps
    params['dataset_train'] = ch_type
    params['dataset_test'] = ch_type
    params['kernel_size'] = kernel_size
    params['timestamp'] = timestamp
    params['trained_epochs'] = train_dict['trained_epochs']
    params['num_min_epochs'] = num_min_epochs
    params['num_epochs_no_improve'] = num_epochs_no_improve
    params['n_layers_pre'] = n_layers_pre
    params['ch_layers_pre'] = ch_layers_pre
    params['n_layers_post'] = n_layers_post
    params['ch_layers_post'] = ch_layers_post
    params['n_layers_time'] = n_layers_time
    params['ch_init_time'] = ch_init_time
    params['num_learnable_params'] = flow_model.num_parameters
    params['fft_pre'] = fft_pre
    params['batch_norm'] = batch_norm
    params['downsamp_fac'] = downsamp_fac
    params['seed'] = seed

    # Create flow_est directory for additional results
    os.makedirs('./results/flow_est/', exist_ok=True)
    file_name = f'./results/flow_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_params.csv'
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
           writer.writerow([key, value])

    # Plot training curves
    file_name = f'./results/flow_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_loss.png'
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss', linewidth=2)
    plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss', linewidth=2)
    plt.legend(['train-loss', 'val-loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Flow Model Training Loss ({ch_type})')
    plt.grid(True, alpha=0.3)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Training curves saved: {file_name}")

    # run testing routine
    if not args.no_test:
        print("\nStarting testing...")
        try:
            test_dict = tester.test()
            print("Testing completed successfully")
        except Exception as e:
            print(f"ERROR: Testing failed: {e}")
            import traceback
            traceback.print_exc()
            return

        if return_all_timesteps:
            # plot all curves
            file_name = f'./results/flow_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                        f'T={num_timesteps}_perstep.png'
            plt.figure(figsize=(10, 6))
            lines = []
            for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
                mse_list_allsteps = test_dict[criteria[0]]['NMSEs_total_power'][isnr]
                snr_now = test_dict[criteria[0]]['SNRs'][isnr]
                n_timesteps_eval = len(mse_list_allsteps)
                lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps, label=f'SNR = {int(snr_now)}')
                plt.xlabel('Timesteps')
                plt.ylabel('nMSE')
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels)
            plt.title(f'Flow Model NMSE vs Timesteps ({ch_type})')
            plt.grid(True, alpha=0.3)
            plt.savefig(file_name, dpi=300, bbox_inches='tight')

            # save all mses
            mse_list = list()
            mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
            mse_list[-1].insert(0, 'SNR')
            mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
            mse_list[-1].insert(0, 'nmse_flow')
            mse_list = [list(i) for i in zip(*mse_list)]
            print(mse_list)
            file_name = f'./results/flow_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}_perstep.csv'
            with open(file_name, 'w') as myfile:
                wr = csv.writer(myfile, lineterminator='\n')
                wr.writerows(mse_list)

            # remove all mses except last to save it later
            for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
                test_dict[criteria[0]]['NMSEs_total_power'][isnr] = test_dict[criteria[0]]['NMSEs_total_power'][isnr][-1]

        mse_list = list()
        mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
        mse_list[-1].insert(0, 'SNR')
        mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
        mse_list[-1].insert(0, 'nmse_flow')
        mse_list = [list(i) for i in zip(*mse_list)]
        print(f"Test results:")
        for row in mse_list:
            print(f"  {row}")
        
        file_name = f'./results/flow_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}.csv'
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(mse_list)

        utils.save_params(dir_result=dir_result, filename='test_results', params=test_dict)
    else:
        print("\nTesting skipped.")
    
    print("\n" + "=" * 60)
    print("FLOW MODEL TRAINING AND TESTING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {dir_result}/train_models/")
    print(f"Results saved to: {dir_result}/")
    print(f"Additional results saved to: ./results/flow_est/")


if __name__ == '__main__':
    main() 