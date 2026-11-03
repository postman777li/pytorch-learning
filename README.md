# PyTorch 学习记录

按天保存简短概念与本人练习，每天两个文件：`notes.md`、`code.py`。

## 计划与进度

目前学习记录明确给出模块一 Day 1–4；PyTorch 阶段的总天数尚未确定，后续按真实计划追加。

| 天数 | 主题与要点 | 文件 | 状态 |
|---|---|---|---|
| Day 1 | Tensor：创建、维度/Shape、索引/切片、广播、矩阵运算、dtype | [概念](day01/notes.md) · [代码](day01/code.py) | 已归档本人练习，运行通过；独立验收未记录 |
| Day 2 | Autograd：计算图、requires_grad、grad_fn、backward、梯度累积、detach、no_grad | [概念](day02/notes.md) · [代码](day02/code.py) | 待学习 |
| Day 3 | nn.Module：__init__、forward、Linear、Sequential、Parameter、parameters、state_dict | [概念](day03/notes.md) · [代码](day03/code.py) | 待学习 |
| Day 4 | 设备与模式：CPU/CUDA、to、train/eval、no_grad、参数/缓冲区、种子与可复现 | [概念](day04/notes.md) · [代码](day04/code.py) | 待学习 |

后续范围：Dataset/DataLoader 与数据处理、Forward/Loss/Backward/Optimizer、SGD/Adam/AdamW、Warmup/Cosine、梯度累积/裁剪/混合精度、验证损失/困惑度、Checkpoint/续训/日志/Debug。尚未分配天数，不预写学习成果。本仓库只记录 PyTorch 阶段。

## 运行

安装 Python 与 [PyTorch](https://pytorch.org/get-started/locally/) 后，在仓库根目录执行：

```bash
python day01/code.py
```

已使用 Python 3.12 / PyTorch 2.11.0+cu128 实际运行 Day1（张量默认在 CPU）。Day2–4 只有注释占位，不代表已实现或验证。

## 来源与后续更新

- 结构、文件名和公开仓库习惯参考 `postman777li/transformer-learning`。
- Day1 `code.py` 原样复制自本地 `tensor_basics.py`，字节一致；原文件留在本地并忽略，避免重复归档。类型转换返回值未接收等细节见概念文件。
- 计划来源：《大模型学习路线规划》《学习Day1张量》。前者给出七阶段方向，后者仅明确模块一四天。
- 每天学习后替换该日占位概念和代码，实际运行，再更新状态并提交。
