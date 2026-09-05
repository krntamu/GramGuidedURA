# GramGuidedURA

Code for Semi-blind Channel Estimation for Unsourced Random Access using Gram-guided Diffusion.

## Dataset

The experiments use the multiuser 3GPP channel dataset.

Download the dataset from:

[Download dataset](https://drive.google.com/drive/folders/1Z00MWbhDOVH_j_ZXETA03wC2EnhroH_d?usp=drive_link)

After downloading, place the dataset files in the `bin/` directory.

## Reproducing the Results

After installing the environment specified in `environment.yml`, run:

```bash
python load_and_eval_dm_dps.py \
  --ch_type pseudo_multiuser_3gpp \
  --model_path "results/2026-03-24-18h35m56s" \
  --method dps_cov_est \
  --exp_key H \
  --pilot_mode nonorthogonal \
  --cov_scale_mode sqrt_beta_t \
  --spatial_pilot_gamma 0 \
  --pilot_power_norm row_norm \
  --dynamic_dps_lambda \
  --dps_lambda_schedule pilot_table \
  --dynamic_cov_lambda \
  --cov_lambda_schedule pilot_table \
  --gaussian_snr_match_mode worst \
  --snr_min -5 \
  --snr_max 30 \
  --snr_step 1 \
  --n_pilot 16 \
  --n_time_samples 2000
````

 The number of pilots can be set to either `16` or `32` using `--n_pilot`.

 The trained model/checkpoint should be placed under:

```
results/2026-03-24-18h35m56s/
```

 ## Acknowledgement

 This repository is based on the code from [Diffusion_channel_est](https://github.com/benediktfesl/Diffusion_channel_est)
