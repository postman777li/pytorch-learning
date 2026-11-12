# PyTorch 与深度学习训练能力

24 天 · 6 个模块 · 每模块 4 天。目标是完成数据加载、训练、优化、验证、保存、恢复与实验分析，最后整合为训练项目。

计划来源为《PyTorch学习规划》，每日概念和代码按学习窗口与本地练习归档。当前已学习并归档 Day1–6，Day7–24 均为待学习占位。每天保留 `notes.md` 和 `code.py`。

## 24 天完整计划

### 模块1 · PyTorch 核心基础（Day 1–4）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 1 | Tensor 核心操作 | Tensor 创建、维度、Shape、索引、切片、广播机制、矩阵运算、数据类型 | `tensor_basics.py`：完成各种 Tensor 操作练习 | [概念](day01/notes.md) · [代码](day01/code.py) | 练习已归档并运行 |
| Day 2 | 自动求导 Autograd | 计算图、requires_grad、grad_fn、反向传播、梯度累积、detach、no_grad | `autograd.py`：手动验证梯度计算过程 | [概念](day02/notes.md) · [代码](day02/code.py) | 练习已归档并运行 |
| Day 3 | nn.Module 模型构建 | nn.Module、`__init__`、forward、nn.Linear、nn.Sequential、Parameter、parameters、state_dict | `model_basics.py`：从零构建一个多层神经网络 | [概念](day03/notes.md) · [代码](day03/code.py) | 教学代码已归档并运行 |
| Day 4 | 设备管理与模型模式 | CPU/GPU、CUDA、to(device)、train()、eval()、no_grad()、模型参数与缓冲区、随机种子、可复现性 | `device_and_mode.py`：完成 CPU/GPU 训练与模式切换实验 | [概念](day04/notes.md) · [代码](day04/code.py) | 练习已归档并运行 |

### 模块2 · 数据处理与输入管线（Day 5–8）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 5 | Dataset | Dataset、`__init__`、`__len__`、`__getitem__`、样本与标签、数据集划分 | `dataset.py`：自己编写一个文本分类数据集 | [概念](day05/notes.md) · [代码](day05/code.py) | 练习已归档并运行 |
| Day 6 | DataLoader | Batch、Batch Size、Shuffle、Sampler、num_workers、pin_memory、drop_last | `dataloader.py`：完成批量数据加载 | [概念](day06/notes.md) · [代码](day06/code.py) | 练习已归档并运行 |
| Day 7 | Tokenization | Tokenizer 工作原理、文本编码、Token ID、Padding、Truncation、Attention Mask | `tokenization.py`：将文本转换为模型输入 | [概念](day07/notes.md) · [代码](day07/code.py) | 待学习 |
| Day 8 | Data Collator | Data Collator、动态 Padding、Batch 内数据整理、输入格式统一、标签处理 | `collator.py`：自己编写一个文本任务的数据整理器 | [概念](day08/notes.md) · [代码](day08/code.py) | 待学习 |

### 模块3 · 模型训练完整流程（Day 9–12）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 9 | Forward & Loss | 前向传播、Logits、预测结果、损失函数、CrossEntropyLoss、MSELoss、标签与预测的对应关系 | `forward_loss.py`：完成模型预测和损失计算 | [概念](day09/notes.md) · [代码](day09/code.py) | 待学习 |
| Day 10 | Backward & Optimizer | loss.backward()、梯度计算、Optimizer、zero_grad()、step()、参数更新 | `backward_optimizer.py`：手动观察一次完整参数更新 | [概念](day10/notes.md) · [代码](day10/code.py) | 待学习 |
| Day 11 | 完整训练循环 | Epoch、Iteration、Batch、Train Loop、训练顺序、梯度清零时机、训练过程中的指标统计 | `train_loop.py`：独立编写完整训练循环 | [概念](day11/notes.md) · [代码](day11/code.py) | 待学习 |
| Day 12 | 验证与泛化 | Train / Validation / Test、数据集划分、过拟合、欠拟合、训练集与验证集表现、Early Stopping | `train_val.py`：训练并验证一个分类模型 | [概念](day12/notes.md) · [代码](day12/code.py) | 待学习 |

### 模块4 · 优化器与学习率调度（Day 13–16）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 13 | SGD 优化器 | SGD、Momentum、梯度下降、学习率对参数更新的影响 | `sgd.py`：手动实现简化版 SGD，并与 PyTorch 对比 | [概念](day13/notes.md) · [代码](day13/code.py) | 待学习 |
| Day 14 | Adam 与 AdamW | 一阶矩、二阶矩、动量、偏差修正、Weight Decay、Adam 与 AdamW 区别 | `adam_adamw.py`：比较三种优化器训练效果 | [概念](day14/notes.md) · [代码](day14/code.py) | 待学习 |
| Day 15 | Learning Rate | Learning Rate、固定学习率、学习率过大与过小、学习率敏感性 | `learning_rate.py`：对比不同学习率下的训练曲线 | [概念](day15/notes.md) · [代码](day15/code.py) | 待学习 |
| Day 16 | Learning Rate Scheduler | Warmup、Linear Decay、Cosine Decay、StepLR、LambdaLR、Scheduler 调用时机 | `lr_scheduler.py`：实现 Warmup + Cosine Decay | [概念](day16/notes.md) · [代码](day16/code.py) | 待学习 |

### 模块5 · 训练优化与稳定性（Day 17–20）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 17 | Gradient Accumulation | 梯度累积、有效 Batch Size、显存限制、Loss 缩放、最后不足一个累积周期的处理 | `gradient_accumulation.py`：比较真实大 Batch 与梯度累积 | [概念](day17/notes.md) · [代码](day17/code.py) | 待学习 |
| Day 18 | Gradient Clipping | 梯度范数、梯度爆炸、clip_grad_norm_、梯度裁剪时机 | `gradient_clipping.py`：观察梯度裁剪前后的变化 | [概念](day18/notes.md) · [代码](day18/code.py) | 待学习 |
| Day 19 | Mixed Precision | FP32、FP16、BF16、autocast、GradScaler、显存与训练速度、数值稳定性 | `mixed_precision.py`：实现混合精度训练 | [概念](day19/notes.md) · [代码](day19/code.py) | 待学习 |
| Day 20 | Training Debug & Stability | Loss NaN、梯度消失、梯度爆炸、过拟合、数据异常、单 Batch 过拟合测试、异常检测、随机种子与排查流程 | `training_debug.py`：人为制造训练问题并定位原因 | [概念](day20/notes.md) · [代码](day20/code.py) | 待学习 |

### 模块6 · 训练工程与综合实战（Day 21–24）

| 天数 | 主题 | 学习内容 | 原计划代码与任务 | 归档 | 状态 |
|---|---|---|---|---|---|
| Day 21 | Checkpoint | 模型保存、加载、state_dict、Optimizer State、Scheduler State、Scaler State、Epoch 和训练进度 | `checkpoint.py`：保存模型并恢复训练状态 | [概念](day21/notes.md) · [代码](day21/code.py) | 待学习 |
| Day 22 | 断点续训与实验日志 | Resume Training、随机数状态恢复、TensorBoard、Loss 曲线、Learning Rate 曲线、实验配置管理、日志记录 | `experiment_logging.py`：实现完整实验记录 | [概念](day22/notes.md) · [代码](day22/code.py) | 待学习 |
| Day 23 | 模型评估 | Train Loss、Validation Loss、Perplexity、Accuracy、Precision、Recall、F1、模型选择、测试集评估 | `evaluation.py`：实现模型评估与指标计算 | [概念](day23/notes.md) · [代码](day23/code.py) | 待学习 |
| Day 24 | 综合实战 | 串联数据处理、模型构建、训练、优化器、Scheduler、AMP、验证、Checkpoint、日志、断点续训 | `full_training_project/`：独立完成一个完整训练项目 | [概念](day24/notes.md) · [代码](day24/code.py) | 待学习 |

### 原计划中的补充重点

| 补充内容 | 为什么需要 | 安排 |
|--------|--------|--------|
| 计算图与梯度累积机制 | 理解自动求导和参数更新的本质 | Day 2、10 |
| GPU 与设备管理 | 实际训练离不开 GPU | Day 4 |
| 模型参数初始化 | 初始化不当会影响训练稳定性 | Day 3、20 |
| Padding 与 Attention Mask | NLP 模型输入处理的基础 | Day 7–8 |
| Train/Eval 模式区别 | Dropout、BatchNorm 等行为不同 | Day 4、12 |
| 数据集划分与数据泄漏 | 防止评估结果失真 | Day 12 |
| 过拟合与正则化 | 理解模型泛化能力 | Day 12、20 |
| 梯度异常排查 | 实际训练中非常常见 | Day 18、20 |
| 随机种子与可复现性 | 保证实验结果可追踪 | Day 4、20 |
| 实验配置管理 | 规范化训练项目的重要组成部分 | Day 22 |
| 评估指标 | 不同任务不能只看 Loss | Day 23 |
| Perplexity | 大语言模型训练的重要评估指标 | Day 23 |

Day24 原计划产出为 `full_training_project/`。当前按每日两个文件的约定保留占位，综合项目在学习当天按实际实现补入。

## 运行

需要 Python、PyTorch，Day4 另需 NumPy。PyTorch 的 CPU/CUDA 安装方式见[官方安装页](https://pytorch.org/get-started/locally/)。

```bash
python day01/code.py
python day02/code.py
python day03/code.py
python day04/code.py
python day05/code.py
python day06/code.py
```

已在 Python 3.12 / PyTorch 2.11.0+cu128 环境运行 Day1–6，Day4 另验证 CPU 回退与 CUDA。生成权重被 Git 忽略。Day7–24 的代码文件仅有待学习注释，概念文件列明原计划主题和任务。

## 来源与整理说明

| 日期目录 | 来源与改动 |
|---|---|
| day01 | 本地 tensor_basics.py 原样归档 |
| day02 | 本地 grad.py 原样归档 |
| day03 | 《学习Day1张量》教学代码，补充各部分例子，权重改存脚本旁 |
| day04 | 本地 device_and_mode.py，修正两处 CUDA 判断漏括号，权重改存脚本旁 |
| day05 | 本地 dataset.py，__len__ 改用实例数据，补上文本/标签数量检查 |
| day06 | 本地 dataloader.py 原样归档 |

Day1–4 概念来自《学习Day1张量》，Day5–6 来自《学习Dataset数据集》，计划来自《PyTorch学习规划》。按学习窗口后续约定，每天提供完整代码与简短概念总结。学习与运行状态如实记录，不以脚本运行代替个人掌握情况。
