# ADRD: Detecting diffusion-generated images via adversarial perturbation induced reconstruction discrepancy

## 项目概述

扩散模型的快速发展引发了对生成欺骗性视觉内容滥用的担忧。现有检测方法多依赖图像语义特征，但现代扩散模型已被优化以密切匹配真实图像的语义结构，降低了基于语义的检测效果。本项目提出 ADRD（对抗扰动诱导重建差异），一个通过在潜在空间引入扰动并测量重建偏差对相同扰动的响应来主动探测重建行为的检测框架。实验表明，真实图像通常表现出更大且更可变的重建响应，而扩散生成的图像倾向于显示更稳定的重建行为。通过刻画重建敏感性而非绝对重建误差，ADRD 为现有重建检测器提供了补充视角。

**主要特点：**
- 🎯 动态响应探测：通过潜在空间扰动主动探测重建行为
- 🚀 快速的图像重建和评估流程
- 📊 支持交叉验证和多运行统计
- 🔧 灵活的配置系统和命令行参数支持

---

## 环境配置

### 系统要求

- **操作系统**：Linux / macOS / Windows
- **Python 版本**：3.9+
- **GPU**：强烈推荐（NVIDIA GPU with CUDA 支持）
- **内存**：至少 16GB RAM（如使用 GPU，显存 8GB+）

### 安装步骤

#### 1. 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n adrd python=3.9
conda activate adrd


#### 2. 安装依赖

```bash
# 安装 requirements.txt 中的所有依赖
pip install -r requirements.txt

# 如果使用 GPU（CUDA 11.x），可能需要手动安装 PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 模型设置

### 下载 Stable Diffusion v1.5 模型

项目需要从 HuggingFace 下载 Stable Diffusion v1.5 的预训练模型组件。

#### 快速下载方法（推荐）

使用 HuggingFace CLI 工具：

```bash
# 安装 huggingface-hub（如果未安装）
pip install huggingface-hub

# 下载整个模型到本地
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --repo-type model \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

#### 手动下载方法

访问 https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

下载以下文件放入 `models/sd_v1_5` 目录：

1. **text_encoder**
   - `config.json`
   - `pytorch_model.bin`

2. **tokenizer**
   - `merges.txt`
   - `special_tokens_map.json`
   - `tokenizer_config.json`
   - `vocab.json`

3. **unet**
   - `config.json`
   - `diffusion_pytorch_model.bin`

4. **vae**
   - `config.json`
   - `diffusion_pytorch_model.bin`

5. **其他文件**
   - `model_index.json`
   - `scheduler/config.json`
   - ...

#### 验证模型

```bash
# 检查模型文件是否完整
ls -la models/sd_v1_5/
```

预期的目录结构：

```
models/sd_v1_5/
├── text_encoder/
│   ├── config.json
│   └── pytorch_model.bin
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
├── model_index.json
└── scheduler/
    └── config.json
```

---

## 快速开始

项目包含 5 个主要步骤的完整工作流程。所有脚本都支持默认配置运行，也可以通过命令行参数进行自定义。

### 步骤 0: 数据集生成（可选）

使用 `generate_dataset.py` 从 GenImage 数据集采样生成 CSV 索引文件。

**脚本位置**：`utils/generate_dataset.py`

**功能**：
- 从 GenImage 数据集中随机选择 AI 生成图像和真实图像
- 生成 CSV 索引文件
- 每次执行产生不同的随机结果

**默认参数运行**：

```bash
python utils/generate_dataset.py
```

**自定义参数运行**：

```bash
python utils/generate_dataset.py \
    --img_root /path/to/GenImage \
    --save_index datasets/GenImage \
    --n 20
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--img_root` | `D:\GenImage` | GenImage 数据集根目录 |
| `--save_index` | `datasets/GenImage` | CSV 输出目录 |
| `--n` | `20` | 每个数据集每个类别的采样数量 |

**输出**：
- `<dataset_name>_ai.csv` - AI 生成图像列表
- `<dataset_name>_nature.csv` - 真实图像列表

---

### 步骤 1: 计算优化 Probe

使用 `quick_optimize.py` 训练潜在空间探针。该脚本进行快速测试，使用最小样本和迭代次数。

**脚本位置**：`quick_optimize.py`

**功能**：
- 加载真实和虚假图像
- 训练通用潜在空间探针
- 计算重建误差间隙
- 保存训练好的探针方向

**默认参数运行**：

```bash
python quick_optimize.py
```

**自定义参数运行**：

```bash
python quick_optimize.py \
    --real datasets/seek_probe/real/wukong_nature.csv \
    --fake datasets/seek_probe/fake/wukong_ai.csv \
    --save_probe results/probes/
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--real` | `datasets/seek_probe/real/wukong_nature.csv` | 真实图像 CSV 文件路径 |
| `--fake` | `datasets/seek_probe/fake/wukong_ai.csv` | 虚假图像 CSV 文件路径 |
| `--save_probe` | `results/probes` | 保存 probe 的目录 |

**配置参数** (在 `configs/config.py` 的 `OptConfig` 中):

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `DEVICE` | auto | 计算设备（cuda 或 cpu） |
| `CHECKPOINT` | `models/sd_v1_5` | 模型检查点路径 |
| `N_SAMPLES_REAL` | `1` | 每个模型的真实图像样本数 |
| `N_SAMPLES_FAKE` | `1` | 每个模型的虚假图像样本数 |
| `N_ITERATIONS` | `20` | 优化迭代次数 |
| `LEARNING_RATE` | `0.1` | 梯度上升学习率 |
| `MC_SAMPLES` | `8` | 蒙特卡洛采样数 |
| `STRENGTH` | `0.6` | 图像引导强度（0.0-1.0） |

**输出**：
- `results/probes/quick_probe.pt` - 训练好的探针方向张量

**预期结果**：
- ✅ 最终间隙（gap）为正，表示真实图像重建误差更大

---

### 步骤 2: 图像重建与探针叠加

使用 `recon.py` 将探针叠加到图像上进行重建，生成用于训练分类器的张量。

**脚本位置**：`recon.py`

**功能**：
- 批量编码图像到潜在空间
- 叠加训练好的探针
- 在潜在空间中重建
- 保存重建差异张量用于后续训练

**默认参数运行**：

```bash
python recon.py
```

**自定义参数运行**：

```bash
python recon.py \
    --recon_target datasets/seek_probe \
    --probe_path data/probe/probe.pt \
    --save_img_tensor results/img_tensor
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--recon_target` | `datasets/seek_probe/fake` | CSV 根目录或单个 CSV 文件 |
| `--probe_path` | `data/probe/probe.pt` | 探针文件路径 |
| `--save_img_tensor` | `results/img_tensor` | 输出张量目录 |

**配置参数** (在 `configs/config.py` 的 `ReconConfig` 中):

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `BATCH_SIZE` | `1` | 批处理大小 |
| `NUM_STEPS` | `25` | 扩散重建步数 |
| `SEED` | `12` | 随机种子 |
| `EPSILON` | `30` | 探针强度 |
| `STRENGTH` | `0.6` | 图像引导强度（0.0-1.0） |

**输出**：
- `results/img_tensor/<arch>/real/*.pt` - 真实图像的重建差异
- `results/img_tensor/<arch>/fake/*.pt` - 虚假图像的重建差异

**文件格式**：
每个 `.pt` 文件包含一个元组 `(tensor, is_fake)`，其中：
- `tensor`: 重建差异张量 (4D: 1×4×64×64)
- `is_fake`: 标签 (0=真实, 1=虚假)

---

### 步骤 3: 训练和验证分类器

使用 `train_eval.py` 进行模型训练和交叉验证评估。

**脚本位置**：`train_eval.py`

**功能**：
- 加载重建的张量数据
- 训练分类器网络
- 进行交叉验证评估
- 计算评估指标（ACC、AP、AUC）
- 保存训练结果

**默认参数运行**：

```bash
python train_eval.py
```

**自定义参数运行**：

```bash
python train_eval.py \
    --train_tensor results/img_tensor \
    --eval_tensor results/img_tensor \
    --save_models results/models \
    --results results/eval_results
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--train_tensor` | `results/img_tensor` | 训练数据目录 |
| `--eval_tensor` | `results/img_tensor` | 评估数据目录 |
| `--save_models` | `results/cross_eval_models/models_GenImage` | 模型保存目录 |
| `--results` | `results/cross_eval_res/models_GenImage` | 结果保存目录 |

**配置参数** (在 `configs/config.py` 的 `TrainConfig` 中):

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `NETWORK` | `BetterCNN` | 网络架构（SimpleCNN/BetterCNN/FusionCNN/AttentionCNN） |
| `EPOCHS` | `50` | 训练轮数 |
| `BATCH_SIZE` | `8` | 批处理大小 |
| `LEARNING_RATE` | `1e-3` | 学习率 |
| `SPLIT_RATIO` | `0.7` | 训练/验证分割比例 |
| `RETRAIN_TIMES` | `3` | 每个子集重复训练次数 |
| `FORCE_RETRAIN` | `False` | 是否强制重新训练 |
| `N_EVAL_RUNS` | `1` | 每个模型的评估运行次数 |

**输出**：
- `results/cross_eval_models/models_GenImage/*.pth` - 训练好的模型
- `results/cross_eval_res/models_GenImage/final_results.json` - 评估结果

**结果格式** (`final_results.json`):

```json
{
  "train_subset_name": {
    "eval_subset_name": {
      "acc": {"mean": 0.75, "std": 0.02},
      "ap": {"mean": 0.68, "std": 0.01},
      "auc": {"mean": 0.66, "std": 0.02}
    }
  }
}
```

---

## 配置详细说明

项目配置在 `configs/config.py` 中定义。主要配置类如下：

### BaseConfig（基础配置）

所有其他配置类的父类，包含全局设置：

```python
class BaseConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "models/sd_v1_5"
    IMAGE_ROOT = "/path/to/GenImage"
    NETWORK = 'BetterCNN'
```

### OptConfig（优化/快速测试配置）

用于 `quick_optimize.py`：

- `REAL_CSV`: 真实图像 CSV 路径
- `FAKE_CSV`: 虚假图像 CSV 路径
- `N_SAMPLES_REAL`: 真实样本数（快速测试：1）
- `N_SAMPLES_FAKE`: 虚假样本数（快速测试：1）
- `N_ITERATIONS`: 优化迭代次数（快速测试：20）
- `LEARNING_RATE`: 学习率（0.1）
- `MC_SAMPLES`: 蒙特卡洛采样数（8）
- `STRENGTH`: 图像引导强度（0.6）

### ReconConfig（重建配置）

用于 `recon.py`：

- `RECON_TARGET`: 重建目标 CSV 目录
- `SAVE_IMG_TENSOR`: 张量输出目录
- `PROBE_PATH`: 探针文件路径
- `NUM_STEPS`: 扩散步数（25）
- `BATCH_SIZE`: 批处理大小（1）
- `EPSILON`: 探针强度（30）
- `STRENGTH`: 图像引导强度（0.6）

### TrainConfig（训练配置）

用于 `train_eval.py`：

- `TRAIN_TENSOR`: 训练数据目录
- `EVAL_TENSOR`: 评估数据目录
- `SAVE_MODELS`: 模型保存目录
- `RESULTS`: 结果保存目录
- `EPOCHS`: 训练轮数（50）
- `BATCH_SIZE`: 批处理大小（8）
- `LEARNING_RATE`: 学习率（1e-3）
- `SPLIT_RATIO`: 训练/验证分割（0.7）
- `RETRAIN_TIMES`: 重复训练次数（3）

---

## 项目目录结构

```
adrd_v3/
├── README.md                    # 本文件
├── requirements.txt             # 项目依赖
│
├── configs/
│   ├── __init__.py
│   └── config.py               # 配置类定义
│
├── core/
│   ├── diffusion_model_v3.py   # 扩散模型核心类
│   ├── optimize_probe.py        # 探针训练算法
│   ├── latent_probe.py          # 探针加载和应用
│   ├── classifier.py            # 分类器网络
│   ├── utilities.py             # 数据加载工具
│   └── utilities_train.py       # 训练工具
│
├── utils/
│   ├── generate_dataset.py      # 数据集生成工具
│   └── minimal_recon.py         # 最小化重建脚本
│
├── analysis/                    # 分析和可视化工具
│   ├── draw.py
│   ├── plot_img_pairs.py
│   ├── plot_matrix.py
│   ├── plot_robust.py
│   └── plot_strength.py
│
├── models/
│   └── sd_v1_5/                # Stable Diffusion v1.5 模型
│
├── datasets/
│   └── seek_probe/              # 数据集 CSV 文件
│       ├── real/
│       │   └── *.csv
│       └── fake/
│           └── *.csv
│
├── data/
│   └── probe/
│       └── probe.pt             # 训练好的探针
│
└── results/                     # 输出结果
    ├── img_tensor/              # 重建张量
    ├── models/                  # 训练好的分类器
    ├── eval_results/            # 评估结果
    └── *.json                   # 结果统计
```

---

## 完整工作流程示例

以下是从数据准备到模型评估的完整工作流程：

### 1. 准备环境

```bash
# 创建虚拟环境
conda create -n adrd python=3.9
conda activate adrd

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 使用 huggingface-cli 下载
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

### 3. 生成数据集（可选）

```bash
python utils/generate_dataset.py \
    --img_root /path/to/dataset \
    --n 20
```

### 4. 训练探针

```bash
python quick_optimize.py
```

### 5. 重建图像

```bash
python recon.py
```

### 6. 训练和评估分类器

```bash
python train_eval.py
```

### 7. 分析结果

```bash
# 查看评估结果
cat results/cross_eval_res/models_GenImage/final_results.json

# 可视化结果（使用分析工具）
python analysis/plot_matrix.py
```

---

## 常见问题和故障排查

### Q1: CUDA 显存不足

**错误提示**：`RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 减少批处理大小
# 在 configs/config.py 中修改：
# ReconConfig.BATCH_SIZE = 1
# TrainConfig.BATCH_SIZE = 4

# 或者使用 CPU（虽然会很慢）
# BaseConfig.DEVICE = torch.device("cpu")
```

### Q2: 模型文件找不到

**错误提示**：`FileNotFoundError: models/sd_v1_5 not found`

**解决方案**：
```bash
# 确保已下载模型
ls -la models/sd_v1_5/

# 如果目录为空，重新下载
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

### Q3: CSV 文件格式错误

**错误提示**：`KeyError: 'imagepath'`

**解决方案**：
- 确保 CSV 文件有 `num`, `imagepath`, `IsFake` 三列
- 使用 `utils/generate_dataset.py` 生成标准格式的 CSV

### Q4: 内存不足

**错误提示**：`MemoryError` 或程序卡死

**解决方案**：
```bash
# 减少样本数量
# 在 configs/config.py 中修改：
# OptConfig.N_SAMPLES_REAL = 1
# OptConfig.N_SAMPLES_FAKE = 1

# 或减少蒙特卡洛采样
# OptConfig.MC_SAMPLES = 4
```

### Q5: 图像加载失败

**错误提示**：`PIL.UnidentifiedImageError` 或图像加载错误

**解决方案**：
- 检查图像路径是否正确
- 确保 `IMAGE_ROOT` 配置指向正确的目录
- 确保 CSV 中的相对路径正确

### Q6: 运行很慢

**可能原因**：
- 使用 CPU 而不是 GPU
- 批处理大小太小
- 扩散步数过多

**解决方案**：
```bash
# 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"

# 增加批处理大小
# 在 configs/config.py 中修改：
# ReconConfig.BATCH_SIZE = 8
# TrainConfig.BATCH_SIZE = 16

# 减少扩散步数
# ReconConfig.NUM_STEPS = 10
```

---


## 引用和许可证

本项目的详细许可证信息请参考 `LICENSE` 文件。

---

## 联系和支持

如有问题或建议，请提交 Issue 或联系项目维护者。
