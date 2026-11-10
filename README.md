# PyTorch 学习记录

按天保存简短概念与本人练习，每天两个文件：`notes.md`、`code.py`。

## 计划与进度

目前学习记录明确给出模块一 Day 1–4；PyTorch 阶段的总天数尚未确定，后续按真实计划追加。

| 天数 | 主题与要点 | 文件 | 状态 |
|---|---|---|---|
| Day 1 | Tensor：创建、维度/Shape、索引/切片、广播、矩阵运算、dtype | [概念](day01/notes.md) · [代码](day01/code.py) | 已归档本人练习，运行通过；独立验收未记录 |
| Day 2 | Autograd：计算图、requires_grad、grad_fn、backward、梯度累积、detach、no_grad | [概念](day02/notes.md) · [代码](day02/code.py) | 已归档本人练习，运行通过；独立验收未记录 |
| Day 3 | nn.Module：__init__、forward、Linear、Sequential、Parameter、parameters、state_dict | [概念](day03/notes.md) · [代码](day03/code.py) | 教学代码已整理并运行；独立验收未记录 |
| Day 4 | 设备与模式：CPU/CUDA、to、train/eval、no_grad、参数/缓冲区、种子与可复现 | [概念](day04/notes.md) · [代码](day04/code.py) | 已归档本人练习并修正设备判断，运行通过；独立验收未记录 |

后续范围：Dataset/DataLoader 与数据处理、Forward/Loss/Backward/Optimizer、SGD/Adam/AdamW、Warmup/Cosine、梯度累积/裁剪/混合精度、验证损失/困惑度、Checkpoint/续训/日志/Debug。尚未分配天数，不预写学习成果。本仓库只记录 PyTorch 阶段。

## 运行

安装 Python 与 [PyTorch](https://pytorch.org/get-started/locally/) 后，在仓库根目录执行：

```bash
python day01/code.py
python day02/code.py
python day03/code.py
python day04/code.py
```

已使用 Python 3.12 / PyTorch 2.11.0+cu128 实际运行 Day1–3（CPU）及 Day4（CPU/CUDA）。Day4 自动选择可用设备，使用 NumPy、PyTorch；权重保存在对应脚本目录并被 Git 忽略。

## 来源与后续更新

- 结构、文件名和公开仓库习惯参考 `postman777li/transformer-learning`。
- Day1 `code.py` 原样复制自本地 `tensor_basics.py`，字节一致；原文件留在本地并忽略，避免重复归档。类型转换返回值未接收等细节见概念文件。
- Day2 `code.py` 原样复制自本地 `grad.py`，字节一致；概念结合学习会话中的梯度累积与 detach/no_grad 补充讲解。原文件留在本地并忽略。
- Day3 按《学习Day1张量》的九个部分提炼重点与例子；代码以会话教学代码为来源，补充概念演示，权重写入脚本旁。没有本地独立练习源文件，不记为本人练习验收通过。
- Day4 代码来自本地 `device_and_mode.py`；修正两处 CUDA 可用性检查漏括号，保存路径改为脚本旁。概念按会话八部分整理重点与例子；原文件留在本地。
- 计划来源：《大模型学习路线规划》《学习Day1张量》。前者给出七阶段方向，后者仅明确模块一四天。
- 每天学习后替换该日占位概念和代码，实际运行，再更新状态并提交。
