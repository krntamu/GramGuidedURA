import numpy as np
import os
from collections import Counter

def generate_pseudo_multiuser_split(input_file: str, in_toeprx: str, in_toeptx: str, 
                                    output_file: str, out_toeprx: str, out_toeptx: str,
                                    meta_file: str, target_samples: int):
    """
    Core function to generate a new dataset split from the original dataset, 
    including proper processing of the Rx/Tx Toeplitz matrices.
    """
    print(f"--- Generating {output_file} ---")
    
    # 1. Load original dataset and Toeplitz matrices
    orig_data = np.load(input_file)
    orig_toeprx = np.load(in_toeprx)
    orig_toeptx = np.load(in_toeptx)
    
    # Sometimes complex data is saved as [N, 64*16] or [N, 1024].
    # Reshape it to [N, 64, 16] if it has only 2 dimensions.
    if len(orig_data.shape) == 2:
        N, total_elements = orig_data.shape
        assert total_elements == 64 * 16, f"Expected 64*16=1024 elements, got {total_elements}"
        # The 3GPP data is stored such that each row has 1024 elements, 
        # representing vec(H) in column-major order (rx varies fastest).
        # We need to reshape using 'F' order so that orig_data[i, j, k] = flat[i, j + 64*k]
        orig_data = np.reshape(orig_data, (N, 64, 16), order='F')
    
    N, rows, cols = orig_data.shape
    assert rows == 64 and cols == 16, f"Expected shape [N, 64, 16], got {orig_data.shape}"
    
    # Prepare arrays for new dataset, toeplitz matrices, and metadata
    new_data = np.zeros((target_samples, rows, cols), dtype=orig_data.dtype)
    new_toeprx = np.zeros((target_samples, cols, rows), dtype=orig_toeprx.dtype)
    new_toeptx = np.zeros((target_samples, cols), dtype=orig_toeptx.dtype)
    
    meta_sample_ids = np.zeros((target_samples, cols), dtype=np.int32)
    meta_col_indices = np.zeros((target_samples, cols), dtype=np.int32)
    meta_permutations = np.zeros((target_samples, cols), dtype=np.int32)
    meta_usage_counts = np.zeros(N, dtype=np.int32)
    
    generated_count = 0
    
    # 3. Repeat multiple shuffle rounds until we reach the desired number of samples
    while generated_count < target_samples:
        # Shuffle all original indices (ensures balance across one full round)
        shuffled_indices = np.random.permutation(N)
        
        # Process in groups of 16
        for i in range(0, N, cols):
            if generated_count >= target_samples:
                break
                
            group_indices = shuffled_indices[i:i+cols]
            
            # We strictly need 16 different samples for one new matrix
            if len(group_indices) < cols:
                continue
                
            new_matrix = np.zeros((rows, cols), dtype=orig_data.dtype)
            selected_cols = np.zeros(cols, dtype=np.int32)
            
            # From each of the 16 matrices, randomly select exactly 1 column
            for j, orig_idx in enumerate(group_indices):
                col_idx = np.random.randint(0, cols)
                new_matrix[:, j] = orig_data[orig_idx, :, col_idx]
                selected_cols[j] = col_idx
                meta_usage_counts[orig_idx] += 1
                
            # Randomly permute the 16 columns once more before saving
            final_perm = np.random.permutation(cols)
            new_matrix = new_matrix[:, final_perm]
            
            # --- TOEPLITZ PROCESSING ---
            # 1. Rx Toeplitz: Save independent Toeplitz vectors for each column (user).
            # The shape will be [16, 64], matching the 16 columns.
            new_toeprx[generated_count] = orig_toeprx[group_indices[final_perm]]
            
            # 2. Tx Toeplitz: Since the 16 users are completely independent, their Tx 
            #    correlation is zero. We construct an Identity matrix Toeplitz vector.
            #    We take the mean of their variances (first element) to maintain scale.
            mean_variance = np.mean(orig_toeptx[group_indices, 0])
            tx_vec = np.zeros(cols, dtype=orig_toeptx.dtype)
            tx_vec[0] = mean_variance
            new_toeptx[generated_count] = tx_vec
            
            # Save the matrix and metadata
            new_data[generated_count] = new_matrix
            
            # We permute the meta records with the same final_perm so that index `c` 
            # correctly describes column `c` of the saved new_matrix.
            meta_sample_ids[generated_count] = group_indices[final_perm]
            meta_col_indices[generated_count] = selected_cols[final_perm]
            meta_permutations[generated_count] = final_perm
            
            generated_count += 1

    # Save outputs
    # Note: If the original data was 2D, we should probably save the new data as 2D as well to match formats.
    # However, since we explicitly reshaped to [N, 64, 16], let's check if we need to flatten it back.
    # Usually datasets like this are loaded via `ut.load_or_create_data`, which might expect [N, 1024].
    # But wait, original data `.shape` might be [N, 64*16] based on the ValueError: "expected 3, got 2".
    
    # We will save the data exactly as it was requested, but flattened if needed.
    # If original was 2D, let's flatten the new data back.
    if len(np.load(input_file).shape) == 2:
        # We need to flatten it back to [N, 1024] such that each row is a column-major vec(H).
        # We can do this by using 'F' order reshape.
        final_data_to_save = np.reshape(new_data, (target_samples, -1), order='F')
    else:
        final_data_to_save = new_data
        
    np.save(output_file, final_data_to_save)
    np.save(out_toeprx, new_toeprx)
    np.save(out_toeptx, new_toeptx)
    np.savez(meta_file, 
             sample_ids=meta_sample_ids, 
             column_indices=meta_col_indices, 
             permutations=meta_permutations, 
             usage_counts=meta_usage_counts)
    
    print(f"Saved {target_samples} samples to {output_file}")
    print(f"Saved Rx/Tx Toeplitz matrices as well.")
    print(f"Saved metadata to {meta_file}\n")


def verify_dataset_split(output_file: str, meta_file: str, expected_samples: int):
    """
    Validation function with all requested sanity checks.
    """
    print(f"--- Verifying {output_file} ---")
    
    new_data = np.load(output_file)
    meta = np.load(meta_file)
    
    sample_ids = meta['sample_ids']
    column_indices = meta['column_indices']
    usage_counts = meta['usage_counts']
    
    # Sanity Check 1: verify each new matrix has shape [64, 16] or [64*16]
    if len(new_data.shape) == 2:
        assert new_data.shape == (expected_samples, 64 * 16), f"Wrong shape: {new_data.shape}"
        print(f"[OK] New data shape is correctly {new_data.shape} (flattened [64*16])")
    else:
        assert new_data.shape == (expected_samples, 64, 16), f"Wrong shape: {new_data.shape}"
        print(f"[OK] New data shape is correctly {new_data.shape}")
    
    # Sanity Check 2: verify no repeated original sample ID inside one new matrix
    for i in range(expected_samples):
        if len(set(sample_ids[i])) != 16:
            raise ValueError(f"Duplicate original sample IDs found in new sample {i}: {sample_ids[i]}")
    print("[OK] No repeated original sample IDs inside any new matrix")
    
    # Sanity Check 3: print usage count statistics for the split
    min_usage = np.min(usage_counts)
    max_usage = np.max(usage_counts)
    avg_usage = np.mean(usage_counts)
    print(f"[OK] Original sample usage counts - Min: {min_usage}, Max: {max_usage}, Avg: {avg_usage:.2f}")
    
    # Sanity Check 4: print column-index usage statistics (0~15)
    col_counts = Counter(column_indices.flatten())
    print("[OK] Column-index selection frequencies (0~15):")
    for c in range(16):
        print(f"  Col {c}: {col_counts.get(c, 0)} times")
    print()


def main():
    # Make output directory for the new dataset
    output_dir = 'bin'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prefix for the generated dataset, to ensure it follows the format expected by load_or_create_data
    prefix = 'pseudo_multiuser_3gpp_path=3'
    orig_prefix = '3gpp_path=3'
    
    splits = [
        {
            'name': 'train',
            'in_file': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=100000_train.npy',
            'in_toeprx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=100000_train_toeprx.npy',
            'in_toeptx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=100000_train_toeptx.npy',
            'out_file': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=100000_train.npy',
            'out_toeprx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=100000_train_toeprx.npy',
            'out_toeptx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=100000_train_toeptx.npy',
            'meta_file': f'{output_dir}/{prefix}_train_meta.npz',
            'target_samples': 100000
        },
        {
            'name': 'verify',
            'in_file': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_val.npy',
            'in_toeprx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_val_toeprx.npy',
            'in_toeptx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_val_toeptx.npy',
            'out_file': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_val.npy',
            'out_toeprx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_val_toeprx.npy',
            'out_toeptx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_val_toeptx.npy',
            'meta_file': f'{output_dir}/{prefix}_val_meta.npz',
            'target_samples': 10000
        },
        {
            'name': 'test',
            'in_file': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_test.npy',
            'in_toeprx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_test_toeprx.npy',
            'in_toeptx': f'bin/{orig_prefix}_dimrx=64_dimtx=16_samp=10000_test_toeptx.npy',
            'out_file': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_test.npy',
            'out_toeprx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_test_toeprx.npy',
            'out_toeptx': f'{output_dir}/{prefix}_dimrx=64_dimtx=16_samp=10000_test_toeptx.npy',
            'meta_file': f'{output_dir}/{prefix}_test_meta.npz',
            'target_samples': 10000
        }
    ]
    
    for s in splits:
        if os.path.exists(s['in_file']):
            generate_pseudo_multiuser_split(s['in_file'], s['in_toeprx'], s['in_toeptx'],
                                            s['out_file'], s['out_toeprx'], s['out_toeptx'],
                                            s['meta_file'], s['target_samples'])
            verify_dataset_split(s['out_file'], s['meta_file'], s['target_samples'])
        else:
            print(f"Input file not found: {s['in_file']}")
            print("Please check the 'in_file' paths.\n")

if __name__ == '__main__':
    main()
