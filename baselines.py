import modules.utils as ut
import csv
import datetime
from estimators.lmmse import LMMSE, mp_eval
import numpy as np
import multiprocessing as mp
import os
import torch
import argparse


def mp_gmm(obj, *args):
    return obj.estimate_from_y(*args)

def mp_omp(obj, *args):
    return obj.estimate(*args)


def main():
    parser = argparse.ArgumentParser(description='Run baseline channel estimation methods')
    parser.add_argument('--device', '-d', default='cuda', type=str, 
                       help='Device to use: cpu or cuda (default: cpu)')
    parser.add_argument('--gpu_batch_size', type=int, default=None,
                       help='Batch size for GPU processing (default: auto-detect based on GPU memory)')
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda':
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            print(f'GPU Memory: {gpu_memory_gb:.1f} GB')
            
            # Auto-detect batch size if not specified
            if args.gpu_batch_size is None:
                # Estimate: each channel needs ~50-60MB, leave 2GB for system/PyTorch overhead
                available_memory_gb = gpu_memory_gb - 2.0
                estimated_batch_size = int(available_memory_gb * 1024 / 60)  # 60MB per channel
                # Clamp to reasonable range
                args.gpu_batch_size = max(10, min(estimated_batch_size, 200))
                print(f'Auto-detected GPU batch_size: {args.gpu_batch_size} (based on {gpu_memory_gb:.1f}GB GPU)')
            else:
                print(f'Using specified GPU batch_size: {args.gpu_batch_size}')
        else:
            print('WARNING: CUDA requested but not available, falling back to CPU')
            device = 'cpu'
    
    if device == 'cpu':
        n_processes = int(mp.cpu_count() / 2)
        print(f'Using CPU with {n_processes} processes')
        # prepare multiprocessing
        pool = mp.Pool(processes=n_processes)
    else:
        pool = None  # Not needed for GPU

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_antennas_rx = 64
    n_antennas_tx = 16
    n_train_ch = 100_000
    n_val_ch = 10_000  # must not exceed size of training set
    n_test_ch = 10_000
    snrs = list(range(-15, 6, 1))  # From -15 to 5, step 1: [-15, -14, ..., 4, 5]
    # ch_type = 'quadriga_LOS'
    ch_type = '3gpp'
    n_path = 3

    eval_LS = False
    eval_lmmse_glob = False
    eval_lmmse_genie = True

    channels_train, toep_train, channels_val, _, channels_test, toep_test = ut.load_or_create_data(ch_type=ch_type,
                            n_path=n_path, n_antennas_rx=n_antennas_rx, n_antennas_tx=n_antennas_tx,
                            n_train_ch=n_train_ch, n_val_ch=n_val_ch, n_test_ch=n_test_ch, return_toep=True)

    # vectorize channels
    n_antennas = n_antennas_rx * n_antennas_tx
    if ch_type.startswith('quadriga'):
        channels_train = np.reshape(channels_train, (-1, n_antennas), 'F')
        channels_test = np.reshape(channels_test, (-1, n_antennas), 'F')

    mse_list = list()
    mse_list.append(snrs.copy())
    mse_list[-1].insert(0, 'SNR')

    if eval_lmmse_glob:
        print('Computing lmmse_glob...')
        mse_list.append(['lmmse_glob'])
        print('  Computing covariance matrix...')
        cov = np.zeros([n_antennas, n_antennas], dtype=complex)
        for i in range(channels_train.shape[0]):
            cov = cov + np.expand_dims(channels_train[i, :], 1) @ np.expand_dims(channels_train[i, :].conj(), 0)
        cov = cov / channels_train.shape[0]
        print('  Evaluating lmmse_glob for all SNRs...')
        if device == 'cpu' and pool is not None:
            eval_list_glob = list()
            for snr in snrs:
                y = ut.get_observation(channels_test, snr)
                eval_list_glob.append([LMMSE(snr), y, cov, False])
            res_glob_lmmse = pool.starmap(mp_eval, eval_list_glob)
        else:
            # CPU sequential (or could also use GPU, but lmmse_glob is fast)
            res_glob_lmmse = []
            for snr in snrs:
                y = ut.get_observation(channels_test, snr)
                res = mp_eval(LMMSE(snr), y, cov, False)
                res_glob_lmmse.append(res)
        for it, res in enumerate(res_glob_lmmse):
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
        print('  lmmse_glob done.')


    if eval_LS:
        print('Computing LS...')
        mse_list.append(['LS'])
        for snr in snrs:
            y = ut.get_observation(channels_test, snr)
            mse_act = np.sum(np.abs(y - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
        print('  LS done.')


    if ch_type == '3gpp' and eval_lmmse_genie:
        print('Computing lmmse_genie...')
        print(f'  Processing {len(snrs)} SNRs × {n_test_ch} test channels = {len(snrs) * n_test_ch} total operations')
        if device == 'cuda':
            print(f'  Using GPU with batch_size={args.gpu_batch_size}')
        else:
            print('  Using CPU multiprocessing')
        mse_list.append(['lmmse_genie'])
        res_genie_lmmse = []
        
        for snr_idx, snr in enumerate(snrs):
            print(f'  Processing SNR={snr}dB ({snr_idx+1}/{len(snrs)})...')
            y = ut.get_observation(channels_test, snr)
            
            if device == 'cuda':
                # Use GPU version with batch processing
                lmmse_obj = LMMSE(snr)
                res = lmmse_obj.estimate_genie_gpu(y, toep_test, device=device, 
                                                    batch_size=args.gpu_batch_size)
            else:
                # Use CPU multiprocessing
                if pool is not None:
                    res = pool.starmap(mp_eval, [[LMMSE(snr), y, toep_test, True]])[0]
                else:
                    res = mp_eval(LMMSE(snr), y, toep_test, True)
            
            res_genie_lmmse.append(res)
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
            print(f'    SNR={snr}dB: MSE={mse_act:.6e}')
        
        print('  lmmse_genie done.')


    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    os.makedirs('./results/baselines/', exist_ok=True)
    file_name = f'./results/baselines/{date_time}_{ch_type}_path={n_path}_ant={n_antennas_rx}x{n_antennas_tx}_' \
                f'testdata={channels_test.shape[0]}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
    
    # Clean up
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()

