# Flow Model 评估指南

## 📋 概述

`load_and_eval_fm.py` 是一个专门用于加载和评估 Flow Model 的脚本，类似于 `load_and_eval_dm.py` 的做法。该脚本支持基于 SNR 的动态步数调整，可以显著提高评估效率。

## 🚀 主要特性

### 1. 自动模型发现
- 自动查找最新的 Flow Model 检查点
- 支持指定特定模型路径
- 智能错误处理和提示

### 2. 动态步数调整
- **自适应步数模式**：根据 SNR 自动调整步数
- **固定步数模式**：使用指定的固定步数
- **基于 DM 结果**：参考 Diffusion Model 的步数进行等比例压缩

### 3. 完整的评估流程
- 多 SNR 测试（-10dB 到 40dB）
- NMSE 性能评估
- 时间统计
- 结果可视化

## 📊 动态步数映射

基于您提供的 DM 步数结果，我们实现了以下映射：

| SNR (dB) | DM 步数 | FM 压缩步数 | 压缩比例 |
|----------|---------|-------------|----------|
| -10      | 68      | 20          | ~29%     |
| -5       | 52      | 15          | ~29%     |
| 0        | 36      | 11          | ~31%     |
| 5        | 23      | 7           | ~30%     |
| 10       | 13      | 4           | ~31%     |
| 15       | 7       | 2           | ~29%     |
| 20       | 4       | 2           | 50%      |
| 25       | 2       | 1           | 50%      |
| 30       | 1       | 1           | 100%     |
| 35       | 1       | 1           | 100%     |
| 40       | 1       | 1           | 100%     |

## 🛠️ 使用方法

### 1. 基本用法

```bash
# 使用自适应步数（推荐）
python load_and_eval_fm.py --adaptive_steps

# 使用固定步数
python load_and_eval_fm.py --fixed_steps 50

# 指定设备
python load_and_eval_fm.py --device cuda --adaptive_steps
```

### 2. 高级选项

```bash
# 指定特定模型
python load_and_eval_fm.py --model_path results/2024-01-01_12-00-00/train_models/model-best.pt --adaptive_steps

# 使用 CPU 和固定步数
python load_and_eval_fm.py --device cpu --fixed_steps 100
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cuda` | 计算设备 (cpu/cuda) |
| `--model_path` | 自动查找 | 指定模型检查点路径 |
| `--adaptive_steps` | False | 启用自适应步数模式 |
| `--fixed_steps` | 100 | 固定步数（当不使用自适应时） |

## 📈 输出结果

### 1. 控制台输出
```
============================================================
FLOW MODEL LOAD AND EVALUATION
============================================================
Device: cuda:0
Adaptive steps: True

Loading test dataset: 3gpp, 10000 samples
Test dataset loaded successfully
Found latest Flow Model: results/2024-01-01_12-00-00/train_models/model-best.pt
Model checkpoint loaded successfully

Creating models...
Models created and loaded successfully
Flow Model timesteps: 100
Number of parameters: 1,234,567

Testing with adaptive steps...
SNR -10 dB: 20 steps
SNR  -5 dB: 15 steps
SNR   0 dB: 11 steps
...
```

### 2. 文件输出

#### CSV 文件
- **详细结果**：`{timestamp}_{ch_type}_dim={n_dim}x{n_dim2}_T={timesteps}_{step_type}_steps_with_time.csv`
- **摘要结果**：`{timestamp}_{ch_type}_dim={n_dim}x{n_dim2}_T={timesteps}_{step_type}_steps.csv`

#### 图表文件
- **性能图**：`{timestamp}_{ch_type}_dim={n_dim}x{n_dim2}_T={timesteps}_{step_type}_steps.png`

### 3. 结果格式

#### 详细结果 CSV
```csv
SNR,nmse_fm,steps,time_s,time_per_sample_ms
-10,0.123456,20,45.67,4.567
-5,0.098765,15,34.56,3.456
0,0.076543,11,25.43,2.543
...
```

#### 摘要结果 CSV
```csv
SNR,nmse_fm,steps
-10,0.123456,20
-5,0.098765,15
0,0.076543,11
...
```

## ⚡ 性能优化

### 1. 自适应步数的优势
- **计算效率**：高 SNR 时使用更少步数，显著减少计算时间
- **性能保持**：在保持性能的同时提高效率
- **智能调整**：根据噪声水平自动优化步数

### 2. 预期性能提升
- **时间节省**：相比固定 100 步，可节省 60-80% 的计算时间
- **资源优化**：减少 GPU 内存使用和计算负载
- **可扩展性**：支持大规模评估任务

## 🔧 故障排除

### 1. 常见问题

#### 找不到模型
```
ERROR: No pre-trained Flow Model found!
Please train a Flow Model first using: python flow_cnn.py
```
**解决方案**：先运行 `python flow_cnn.py` 训练模型

#### CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：减少 batch_size 或使用 CPU

#### 数据集加载失败
```
ERROR: Failed to load dataset
```
**解决方案**：检查 `bin` 目录中是否有数据集文件

### 2. 调试模式

```bash
# 使用小数据集快速测试
python load_and_eval_fm.py --fixed_steps 10 --device cpu
```

## 📝 使用建议

### 1. 推荐配置
```bash
# 生产环境评估
python load_and_eval_fm.py --device cuda --adaptive_steps

# 快速测试
python load_and_eval_fm.py --device cuda --fixed_steps 50

# 详细分析
python load_and_eval_fm.py --device cuda --adaptive_steps --model_path path/to/specific/model.pt
```

### 2. 结果分析
- 比较不同步数策略的性能
- 分析计算时间与精度的权衡
- 验证自适应步数的有效性

## 🎯 预期结果

使用自适应步数时，您应该看到：
- **NMSE 性能**：与固定步数相当或略优
- **计算时间**：显著减少（特别是高 SNR 时）
- **步数分布**：随 SNR 增加而减少
- **资源使用**：更高效的 GPU 利用率

现在您可以使用这个脚本进行高效的 Flow Model 评估了！🎉 