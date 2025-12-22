# Flow Model 训练指南

## 快速开始

### 1. 基本训练
```bash
cd Diffusion_channel_est
python flow_cnn.py --device cuda
```

### 2. 快速测试模式
```bash
python flow_cnn.py --device cuda --quick --epochs 10
```

### 3. 自定义参数训练
```bash
python flow_cnn.py --device cuda --epochs 50 --batch_size 64 --lr 5e-5 --timesteps 50
```

### 4. 自定义早停参数
```bash
python flow_cnn.py --device cuda --epochs 100 --min_epochs 15 --patience 8
```

### 5. 仅训练不测试
```bash
python flow_cnn.py --device cuda --no_test
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | cuda | 训练设备 (cpu/cuda) |
| `--epochs` | 100 | 训练轮数（已优化） |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--timesteps` | 100 | 时间步数 |
| `--samples` | 100000 | 训练样本数 |
| `--quick` | False | 快速训练模式 |
| `--min_epochs` | 10 | 早停前最小训练轮数 |
| `--patience` | 10 | 无改善时的等待轮数 |
| `--no_test` | False | 跳过训练后的测试 |

## 早停机制

Flow Model 训练使用智能早停机制：

- **最小训练轮数**: 确保模型有足够的训练时间
- **耐心值**: 连续多少轮验证损失无改善后停止
- **自动停止**: 避免过度训练，节省时间和计算资源

### 早停示例
```
Epoch 25/100: val_loss = 0.5011505484580994 
Epoch 26/100: val_loss = 0.5011497139930725 
Epoch 27/100: val_loss = 0.501152515411377 
Epoch 28/100: val_loss = 0.5011638402938843 
Early stopping. End of training.
```

## 训练输出

训练完成后，您将得到：

### 1. 模型文件
```
results/[timestamp]/train_models/model-[checkpoint].pt
```

**模型保存机制：**
- 自动保存：每次验证时自动保存模型
- 保存模式：支持 'best'（最佳模型）、'newest'（最新模型）、'all'（所有检查点）
- 默认模式：'best' - 只保存验证损失最低的模型
- 强制保存：训练完成时强制保存最终模型

### 2. 训练结果
```
results/[timestamp]/
├── sim_params.json          # 训练参数
├── train_results.json       # 训练结果
└── test_results.json        # 测试结果
```

### 3. 额外结果
```
results/flow_est/
├── [timestamp]_[config]_params.csv    # 参数总结
├── [timestamp]_[config]_loss.png      # 训练曲线
└── [timestamp]_[config].csv           # 测试结果
```

## 测试训练好的模型

```bash
python test_matrix_denoising.py
```

测试脚本会自动：
1. 查找最新的 Flow Model 检查点
2. 加载训练好的权重
3. 执行矩阵降噪测试
4. 比较原始降噪和条件降噪方法

## 快速训练模式

使用 `--quick` 参数可以快速验证训练流程：

- 减少训练样本到 10,000
- 减少验证样本到 1,000
- 限制训练轮数到 30
- 减少批次大小到 64
- 最小训练轮数：5
- 耐心值：5

## Flow Model 特定参数

Flow Model 使用 EDM (Elucidating Diffusion Models) 框架：

- `sigma_min`: 0.01 (最小噪声水平)
- `sigma_max`: 50.0 (最大噪声水平)
- `rho`: 7.0 (噪声调度参数)
- `sampling_eps`: 0.002 (采样精度)

## 监控训练

训练过程中会显示：
- 训练损失和验证损失
- 模型参数数量
- 训练进度
- 最佳模型保存信息
- 模型文件验证信息
- 早停信息

## 完成训练

训练完成后，您将看到：
```
============================================================
FLOW MODEL TRAINING AND TESTING COMPLETED
============================================================
Model saved to: results/[timestamp]/train_models/
Results saved to: results/[timestamp]/
Additional results saved to: ./results/flow_est/
Model files found: ['model-1.pt', 'model-5.pt', ...]
  model-1.pt: 45.2 KB
  model-5.pt: 45.2 KB
```

## 训练时间优化

- **默认设置**: 100 epochs，通常 20-30 epochs 就能收敛
- **早停机制**: 自动检测收敛，避免过度训练
- **快速模式**: 适合快速验证和调试
- **GPU 加速**: 默认使用 CUDA，显著提升训练速度

现在您可以使用训练好的 Flow Model 进行矩阵降噪了！ 