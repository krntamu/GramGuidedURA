#!/bin/bash
# Debug script to test three cov_scale_mode settings

SNR_DB=-10.0  # Fixed SNR for comparison
COV_LAMBDA=0.01
COV_GRAD_NORM=global
COV_STEP_CLIP=2.0

echo "============================================================"
echo "Cov Scaling Debug Test"
echo "============================================================"
echo "SNR: ${SNR_DB} dB"
echo "cov_lambda: ${COV_LAMBDA}"
echo "cov_grad_norm: ${COV_GRAD_NORM}"
echo "cov_step_clip: ${COV_STEP_CLIP}"
echo ""

# Test 1: constant
echo "----------------------------------------"
echo "Test 1: cov_scale_mode=constant"
echo "----------------------------------------"
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda ${COV_LAMBDA} \
    --cov_scale_mode constant \
    --cov_grad_norm ${COV_GRAD_NORM} \
    --cov_step_clip ${COV_STEP_CLIP} \
    --debug_cov_scaling \
    --sanity_snrs \
    --dps_lambda 0.1 \
    2>&1 | tee results/dm_dps/debug_cov_scaling/test1_constant.log

echo ""
echo ""

# Test 2: sqrt_beta_t
echo "----------------------------------------"
echo "Test 2: cov_scale_mode=sqrt_beta_t"
echo "----------------------------------------"
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda ${COV_LAMBDA} \
    --cov_scale_mode sqrt_beta_t \
    --cov_grad_norm ${COV_GRAD_NORM} \
    --cov_step_clip ${COV_STEP_CLIP} \
    --debug_cov_scaling \
    --sanity_snrs \
    --dps_lambda 0.1 \
    2>&1 | tee results/dm_dps/debug_cov_scaling/test2_sqrt_beta_t.log

echo ""
echo ""

# Test 3: beta_t
echo "----------------------------------------"
echo "Test 3: cov_scale_mode=beta_t"
echo "----------------------------------------"
python load_and_eval_dm_dps.py \
    --method dps_cov_oracle \
    --cov_lambda ${COV_LAMBDA} \
    --cov_scale_mode beta_t \
    --cov_grad_norm ${COV_GRAD_NORM} \
    --cov_step_clip ${COV_STEP_CLIP} \
    --debug_cov_scaling \
    --sanity_snrs \
    --dps_lambda 0.1 \
    2>&1 | tee results/dm_dps/debug_cov_scaling/test3_beta_t.log

echo ""
echo "============================================================"
echo "All tests completed!"
echo "Check CSV files in: results/dm_dps/debug_cov_scaling/"
echo "============================================================"

