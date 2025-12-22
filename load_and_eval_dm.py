"""
Train and test script for the DMCE.
"""
import os
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import DMCE
import torch
from DMCE.utils import cmplx2real

CUDA_DEFAULT_ID = 0


def main():
    parser = argparse.ArgumentParser()
    # Auto-detect device: use CUDA if available, otherwise CPU
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', '-d', default=default_device, type=str)
    parser.add_argument(
        '--ch_type',
        type=str,
        default='3gpp',
        choices=['3gpp', 'quadriga_LOS'],
        help='Channel type: "3gpp" (default) or "quadriga_LOS". '
             'Must match the available best_models_dm_paper subfolder.',
    )
    parser.add_argument(
        '--n_path',
        type=int,
        default=3,
        help='Number of paths for 3gpp channel (ignored for quadriga_LOS).',
    )

    # get the used device
    args = parser.parse_args()
    device = args.device
    
    # Print device information
    print(f"\n{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_dim = 64 # RX antennas
    n_dim2 = 16 # TX antennas
    num_train_samples = 100_000
    num_val_samples = 10_000  # must not exceed size of training set
    num_test_samples = 10_000

    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = True # learn channel distribution in angular domain through Fourier transform
    reverse_add_random = False # re-sampling in the reverse process

    # set data params
    ch_type = args.ch_type  # {'quadriga_LOS', '3gpp'}
    n_path = args.n_path
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'

    _, _, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                                             n_antennas_tx=n_dim2, n_train_ch=num_train_samples,
                                                             n_val_ch=num_val_samples,
                                                             n_test_ch=num_test_samples, return_toep=False)
    del _
    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()
    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # load the model parameter dictionaries
    cwd = os.getcwd()
    #which_dataset = dataset
    model_dir = os.path.join(cwd, './results/best_models_dm_paper', ch_type)
    sim_params = DMCE.utils.load_params(os.path.join(model_dir, 'sim_params'))
    cnn_dict = sim_params['unet_dict']
    diff_model_dict = sim_params['diff_model_dict']

    # manually set the correct device for this simulation
    cnn_dict['device'] = device

    # instantiate the neural network
    cnn = DMCE.CNN(**cnn_dict)

    # instantiate the diffusion model and give it a reference to the unet model
    diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)

    # load the parameters of the pre-trained model into the DiffusionModel instance
    model_path = os.path.join(model_dir, 'train_models')
    model_list = os.listdir(model_path)
    model_path = os.path.join(model_path, model_list[-1])
    model_params = torch.load(model_path, map_location=device)

    diffusion_model.load_state_dict(model_params['model'])

    print("T =", diffusion_model.num_timesteps)
    print("len(betas) =", diffusion_model.betas.numel())
    print("diff_model_dict =", sim_params['diff_model_dict'])

    # Tester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': 512,
        'criteria': ['nmse'],
        'complex_data': False,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
    }

    # instantiate the Tester and give it a reference to the diffusion model as well as testing data
    tester = DMCE.Tester(diffusion_model, data=data_test, **tester_dict)

    num_timesteps = sim_params['diff_model_dict']['num_timesteps']

    diffusion_model.reverse_add_random = reverse_add_random

    # Custom test function with SNR range: -15 to 5 dB, step 1
    def custom_test_nmse():
        """Custom _test_nmse that tests SNR -15 to 5 dB (step 1)"""
        import time
        from tqdm import tqdm
        from DMCE import functional
        import modules.utils as ut
        
        # specify which SNRs should be evaluated (in dB) - range: -15 to 5 dB, step 1
        snr_db_range = torch.arange(-15, 6, 1, dtype=torch.float32, device=tester.device)
        # corresponding linear SNR ρ = 10^(SNR/10)
        snr_range = 10 ** (snr_db_range / 10)
        
        nmse_total_power_list = []
        timings_sec = []
        steps_list = []
        tps_ms_list = []
        
        with torch.no_grad():
            snr_db_list = snr_db_range.tolist()
            for snr_idx, snr in enumerate(tqdm(iterable=snr_range, desc="SNR sweep")):
                snr_db = snr_db_list[snr_idx]
                t_hat = int(torch.abs(tester.model.snrs - snr).argmin())
                steps_list.append(int(t_hat))
                
                print(f"\n[SNR {snr_db:.1f} dB] Processing baseline DM...")
                
                # timing start
                if tester.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                x_hat = []
                for batch_idx, data_batch in enumerate(tester.dataloader):
                    try:
                        data_batch = data_batch.to(device=tester.device)
                        y = functional.awgn(data_batch, snr, multiplier=tester.model.noise_multiplier)
                        
                        x_est = tester.model.generate_estimate(
                            y.to(device=tester.device), snr,
                            return_all_timesteps=tester.return_all_timesteps,
                        )
                        
                        if tester.fft_pre:
                            if tester.return_all_timesteps:
                                x_est = ut.complex_1d_fft(x_est, ifft=True, mode=tester.mode, _4d_array=True)
                            else:
                                x_est = ut.complex_1d_fft(x_est, ifft=True, mode=tester.mode)
                        
                        x_hat.append(x_est.cpu())
                        
                        # Clean up GPU memory
                        del x_est, y, data_batch
                        if tester.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        raise
                
                x_hat = torch.cat(x_hat, dim=0)
                
                # timing end
                if tester.device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                timings_sec.append(dt)
                tps_ms_list.append(dt * 1000.0 / tester.num_samples)
                
                # compute NMSE
                if len(tester.data.shape) == 4:
                    dim = int(tester.data.shape[-1] * tester.data.shape[-2])
                    x_hat_rs = ut.reshape_fortran(x_hat, (-1, dim))
                    nmse_total_power_list.append(
                        functional.nmse_torch(
                            ut.reshape_fortran(torch.squeeze(tester.data), (-1, dim)),
                            x_hat_rs,
                            norm_per_sample=False
                        )
                    )
                else:
                    nmse_total_power_list.append(
                        functional.nmse_torch(
                            torch.squeeze(tester.data),
                            torch.squeeze(x_hat),
                            norm_per_sample=False
                        )
                    )
        
        return {
            'SNRs': snr_db_range.tolist(),
            'NMSEs_total_power': nmse_total_power_list,
            'Steps': steps_list,
            'Timings_sec': timings_sec,
            'Time_per_sample_ms': tps_ms_list,
        }
    
    # Replace the test function to use custom SNR range
    print(f"\n{'='*60}")
    print(f"Testing DM with SNR range: -15 to 5 dB (step 1)")
    print(f"{'='*60}\n")
    
    # Update both _test_nmse and test_funcs list
    tester._test_nmse = custom_test_nmse
    if 'nmse' in tester.criteria:
        nmse_idx = tester.criteria.index('nmse')
        tester.test_funcs[nmse_idx] = custom_test_nmse
    
    # call the test() function. This returns a dictionary with the testing stats.
    # Depending on the size of the test set, this might take a while.
    test_dict = tester.test()

    os.makedirs('./results/dm_est/', exist_ok=True)

    # 汇总（只保留最终一步的 NMSE）
    snrs = test_dict['nmse']['SNRs'].copy()
    nmses = test_dict['nmse']['NMSEs_total_power'].copy()
    steps = test_dict['nmse'].get('Steps', [None]*len(snrs))
    times = test_dict['nmse'].get('Timings_sec', [None]*len(snrs))
    tps   = test_dict['nmse'].get('Time_per_sample_ms', [None]*len(snrs))

    table = []
    table.append(snrs); table[-1].insert(0, 'SNR')
    table.append(nmses); table[-1].insert(0, 'nmse_dm')
    table.append(steps); table[-1].insert(0, 'steps')
    table.append(times); table[-1].insert(0, 'time_s')
    table.append(tps);   table[-1].insert(0, 'time_per_sample_ms')
    rows = [list(row) for row in zip(*table)]

    print(rows)  # 控制台也打印一份

    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_resamp={reverse_add_random}_best_with_time.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(rows)


    if return_all_timesteps:
        # plot all curves
        file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                    f'T={num_timesteps}_perstep_best.png'
        plt.figure()
        lines = []
        for isnr in range(len(test_dict['nmse']['NMSEs_total_power'])):
            mse_list_allsteps = test_dict['nmse']['NMSEs_total_power'][isnr]
            snr_now = test_dict['nmse']['SNRs'][isnr]
            n_timesteps_eval = len(mse_list_allsteps)
            lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps,
                                  label=f'SNR = {int(snr_now)}')
            #plt.legend([f'SNR = {int(snr_now)}'])
            plt.xlabel('Timesteps')
            plt.ylabel('nMSE')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.savefig(file_name)

        # save all mses
        list_snrs_all = test_dict['nmse']['SNRs'].copy()
        list_mses_all = test_dict['nmse']['NMSEs_total_power'].copy()
        for i in range(len(list_snrs_all)):
            n_timesteps_eval = len(list_mses_all[i])
            mse_list = list()
            mse_list.append(list(range(n_timesteps_eval))[::-1])
            mse_list[-1].insert(0, 't')
            mse_list.append(list_mses_all[i])
            mse_list[-1].insert(0, 'nmse_dm')
            mse_list = [list(i) for i in zip(*mse_list)]
            file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                        f'T={num_timesteps}_best_SNR={list_snrs_all[i]}.csv'
            with open(file_name, 'w') as myfile:
                wr = csv.writer(myfile, lineterminator='\n')
                wr.writerows(mse_list)

        # remove all mses except last to save it later
        for isnr in range(len(test_dict['nmse']['NMSEs_total_power'])):
            test_dict['nmse']['NMSEs_total_power'][isnr] = test_dict['nmse']['NMSEs_total_power'][isnr][-1]

    mse_list = list()
    mse_list.append(test_dict['nmse']['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict['nmse']['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_resamp={reverse_add_random}_best.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)


if __name__ == '__main__':
    main()