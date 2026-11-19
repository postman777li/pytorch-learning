# Day 20：Training Debug & Stability（训练排查与稳定性）

今天是模块五的最后一天。相比前几天，今天不是学习某一个新的训练技巧，而是学习一种非常重要的工程能力：**当模型训练出现问题时，如何独立定位问题、分析原因并解决问题。**

继续沿用你的 FruitClassifier 模型，在 Day 17–19 的代码基础上进行改造。

## 一、具体例子：模型训练不正常怎么办？

假设你正在训练一个大模型，遇到了以下几种情况：

**训练集很好，验证集很差**

| 阶段 | Train Loss | Val Loss | 判断 |
|---|---|---|---|
| 训练前期 | 下降 | 下降 | 两者都在改善 |
| 后续训练 | 继续下降 | 开始回升 | 可能过拟合，结合验证指标判断 |

可能出现过拟合

初学者遇到这些问题，通常会直接修改学习率、增加训练轮数或者更换优化器。

但真正的工程排查不是盲目调参，而是沿着训练流程逐步检查，找到问题最早出现的位置。

---

## 二、建立训练排查的整体思路

一个模型的训练，本质上就是下面这条链路：

> 原窗口链路图未返回，以下按本课训练流程补充。

```mermaid
flowchart LR
    A[输入数据与标签] --> B[前向传播 Logits]
    B --> C[Loss]
    C --> D[反向传播]
    D --> E[还原并检查梯度]
    E --> F[梯度裁剪与参数更新]
    F --> G[训练与验证指标]
```

**哪里最先出现异常，就优先排查哪里。**

接下来我们按照这条链路学习具体的排查方法。

---

## 三、问题一：Loss 变成 NaN

### 1\. 什么是 NaN 和 Inf？

| 数值 | 含义 | 常见原因 |
|--------|--------|--------|
| `NaN` | Not a Number，无效数值 | 0 除以 0、负数开平方 |
| `Inf` | 无穷大 | 数值溢出、非零数除以零 |
| `-Inf` | 负无穷大 | 例如计算 log(0) |

如果模型中某一步产生 NaN，并且继续进行反向传播和参数更新，就可能污染模型参数，导致之后的训练全部异常。

### 2\. 如何检测？

PyTorch 已经封装好了三个函数：

```python
torch.isnan(x)     # 检查是否存在 NaN
torch.isinf(x)     # 检查是否存在 Inf
torch.isfinite(x)  # 检查是否为有限数值
```

例如：

```python
x = torch.tensor([1.0, float("nan"), float("inf")])

print(torch.isfinite(x))
```

输出：

```text
tensor([ True, False, False])
```

如果我们想检查整个张量是否正常：

```python
if not torch.isfinite(x).all():
    raise RuntimeError("发现异常数值")
```

其中：

- `.all()`：是否所有元素都是 True。
- `not`：取反。
- `raise RuntimeError()`：主动抛出异常，终止程序。

### 3\. 应用到训练循环

```python
logits = model(batch_x)

if not torch.isfinite(logits).all():
    raise RuntimeError("Logits 出现 NaN 或 Inf")

loss = criterion(logits, batch_y)

if not torch.isfinite(loss):
    raise RuntimeError("Loss 出现 NaN 或 Inf")
```

这样就可以判断，异常究竟是在前向传播中产生的，还是在损失计算中产生的。

---

## 四、问题二：梯度爆炸与梯度消失

Day 18 已经讲过梯度爆炸，所以今天重点学习如何检测。

### 1\. 检查梯度范数

```python
grad_norm = nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True
)
```

相比 Day 18，今天新增了：

```python
error_if_nonfinite=True
```

意思是：如果梯度总范数为 NaN 或 Inf，就直接抛出异常。

这样可以避免程序继续使用异常梯度更新参数。

### 2\. 梯度消失是什么？

与梯度爆炸相反，梯度消失是指梯度在反向传播过程中变得极小，导致部分参数难以有效更新。

例如一个深层模型：

反向传播方向：输出层 → 输入层

示意：越靠近输入层，梯度越小。条形长度采用示意尺度。

### 3\. 如何检查每一层的梯度？

```python
for name, param in model.named_parameters():

    if param.grad is not None:

        grad_norm = param.grad.norm().item()

        print(name, grad_norm)
```

这里使用了一个新函数：

```python
model.named_parameters()
```

之前你使用的 `model.parameters()` 只返回参数，而 `named_parameters()` 会同时返回参数名称和参数本身。

例如：

| name | 参数 Shape |
|--------|--------|
| `network.0.weight` | `[16, 2]` |
| `network.0.bias` | `[16]` |
| `network.2.weight` | `[3, 16]` |
| `network.2.bias` | `[3]` |

通过它，我们就能发现究竟是哪一层梯度异常。

注意：梯度小不一定意味着梯度消失。模型接近收敛时，梯度也可能很小，需要结合 Loss、训练阶段和各层梯度共同判断。

---

## 五、问题三：数据异常

很多时候模型训练不正常，根本不是模型的问题，而是数据有问题。

对于之前的 FruitClassifier，我们期望：

| 检查对象 | 正确情况 |
|--------|--------|
| `batch_x.shape` | `[B, 2]` |
| `batch_y.shape` | `[B]` |
| `batch_x.dtype` | `torch.float32` |
| `batch_y.dtype` | `torch.long` |
| 标签取值 | 0、1、2 |
| 输入数据 | 不包含 NaN 和 Inf |

可以直接使用 `assert` 检查：

```python
assert batch_x.ndim == 2
assert batch_x.shape[1] == 2

assert batch_y.ndim == 1
assert batch_y.dtype == torch.long

assert batch_y.min() >= 0
assert batch_y.max() < 3
```

`assert` 的意思是：如果条件不成立，就抛出 `AssertionError`。

例如：

```python
assert batch_y.max() < 3, "标签超出类别范围"
```

如果三分类任务中出现了标签 5，程序就会直接报错。

在正式工程中，对于不能省略的检查，通常使用 `if + raise`，因为 Python 的优化模式可以禁用 `assert`。

---

## 六、问题四：Loss 一直不下降

如果数据、Logits、Loss、梯度均没有明显异常，但是模型就是学不会，应该怎么办？

这里介绍一个非常实用的工程技巧。

### Single Batch Overfitting（单 Batch 过拟合测试）

假设整个数据集有 100 万条样本，模型训练不正常，你不需要反复拿这 100 万条数据进行测试。

只取出一个 Batch，例如 32 条样本，然后让模型反复学习这 32 条样本。

```python
batch_x, batch_y = next(iter(loader))
```

这里的语法需要理解：

```python
iter(loader)       # 创建 DataLoader 迭代器
next(iter(loader)) # 取出第一个 Batch
```

然后反复训练：

```python
for step in range(200):

    optimizer.zero_grad()

    logits = model(batch_x)

    loss = criterion(logits, batch_y)

    loss.backward()

    optimizer.step()
```

注意，这里没有遍历 DataLoader，而是反复使用**同一批数据**。

单 Batch 过拟合测试：预期现象

示意曲线，并非实际运行结果。

如果一个容量足够的模型，连固定的几十条样本都无法拟合，就应该优先检查训练代码、标签、模型结构、学习率等问题。

如果能成功拟合，说明基本训练链路大概率是有效的，但不代表整个训练系统一定没有其他问题。

**单 Batch 过拟合不是为了得到一个好模型，而是为了测试训练系统是否具备正常学习的能力。**

---

## 七、问题五：训练集很好，验证集很差

这个问题在 Day 12 已经学过，这里简单回顾即可。

| Train Loss | Val Loss | 可能的情况 |
|--------|--------|--------|
| 持续下降 | 持续下降 | 正常学习 |
| 很高 | 很高 | 欠拟合或训练异常 |
| 持续下降 | 开始上升 | 过拟合 |
| 剧烈波动 | 剧烈波动 | 学习率过大或训练不稳定 |

对于过拟合，可以考虑增加训练数据、正则化、Dropout、Weight Decay、Early Stopping 等。

因此，实际工程中不能只记录 Train Loss，还应该记录 Validation Loss。

---

## 八、另外两个实用的 Debug 工具

### 1\. Autograd 异常检测

PyTorch 提供了一个专门用于排查反向传播异常的工具：

```python
with torch.autograd.detect_anomaly():

    logits = model(batch_x)

    loss = criterion(logits, batch_y)

    loss.backward()
```

当反向传播产生 NaN 等异常时，它可以辅助定位是哪一个前向运算导致的。

但是，这个功能会明显降低训练速度。

**只在 Debug 时开启，正常训练时关闭。**

### 2\. 固定随机种子

这个你之前一直在使用：

```python
torch.manual_seed(42)
```

更加完整的写法：

```python
import random
import numpy as np
import torch

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

为什么 Debug 时需要固定随机种子？

因为我们希望修改代码前后，尽量保持模型初始化、数据顺序等条件一致，以便判断问题是否真的得到解决。

不过，固定随机种子并不能保证不同硬件、PyTorch 版本和所有 GPU 算子完全可复现。

---

## 九、Day 20 完整代码：training\_debug.py

今天的完整代码保留 Day 17 的梯度累积、Day 18 的梯度裁剪以及 Day 19 的混合精度，并新增：

- 数据与数值异常检测。
- 梯度监控。
- 单 Batch 过拟合测试。
- 训练集与验证集评估。
- 人为制造异常，验证排查机制。

为了方便你学习，我专门设置了 `fault_mode`，通过修改一个变量就可以模拟不同的训练问题。

```python

import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)
from contextlib import nullcontext

# ============================================
# 1. Day20：固定随机种子
# ============================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================
# 2. Day20：Debug 配置
# ============================================

debug_mode = True

# 可选：
# "normal"        正常训练
# "nan_input"     输入数据包含 NaN
# "bad_label"     标签越界
# "gradient_big"  人为放大梯度
# "gradient_zero" 人为制造零梯度

fault_mode = "normal"

# 是否执行单 Batch 过拟合测试
single_batch_test = False

# 是否开启 Autograd 异常检测
detect_anomaly = False

# ============================================
# 3. 设备与混合精度（Day19）
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"

if use_amp and torch.cuda.is_bf16_supported():

    amp_dtype = torch.bfloat16

else:

    amp_dtype = torch.float16

use_scaler = (
    use_amp and amp_dtype == torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_scaler
)

print("当前设备:", device)

# ============================================
# 4. 创建合成数据
# ============================================

num_per_class = 100

class_0 = (
    torch.randn(num_per_class, 2) * 0.25
    + torch.tensor([1.0, 1.0])
)

class_1 = (
    torch.randn(num_per_class, 2) * 0.25
    + torch.tensor([3.0, 1.0])
)

class_2 = (
    torch.randn(num_per_class, 2) * 0.25
    + torch.tensor([2.0, 3.0])
)

x = torch.cat(
    [class_0, class_1, class_2],
    dim=0
)

y = torch.cat([
    torch.zeros(num_per_class, dtype=torch.long),
    torch.ones(num_per_class, dtype=torch.long),
    torch.full((num_per_class,), 2, dtype=torch.long)
])

# ============================================
# 5. Day20：人为制造数据异常
# ============================================

if fault_mode == "nan_input":

    x[0, 0] = float("nan")

elif fault_mode == "bad_label":

    y[0] = 5

# ============================================
# 6. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

# Day20：划分训练集和验证集

train_size = int(len(dataset) * 0.8)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ============================================
# 7. 定义模型
# ============================================

class FruitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# ============================================
# 8. 训练参数
# ============================================

num_epochs = 10

# Day17：梯度累积
accumulation_steps = 4

effective_batch_size = (
    train_loader.batch_size
    * accumulation_steps
)

# Day18：梯度裁剪
max_norm = 1.0

# ============================================
# 9. Day20：数据检查函数
# ============================================

def check_batch(batch_x, batch_y):

    assert batch_x.ndim == 2
    assert batch_x.shape[1] == 2

    assert batch_y.ndim == 1
    assert batch_x.size(0) == batch_y.size(0)

    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.long

    if not torch.isfinite(batch_x).all():
        raise RuntimeError("输入数据存在 NaN 或 Inf")

    if batch_y.min() < 0 or batch_y.max() >= 3:
        raise RuntimeError("标签超出类别范围")

# 在训练开始前检查整个数据集
if debug_mode:
    check_batch(x, y)

# ============================================
# 10. Day20：梯度检查函数
# ============================================

def check_gradients(model, print_details=False):

    has_gradient = False

    for name, param in model.named_parameters():

        if param.grad is None:

            if print_details:
                print(f"{name}: 没有梯度")

            continue

        has_gradient = True

        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"{name} 的梯度存在 NaN 或 Inf"
            )

        if print_details:

            grad_norm = param.grad.norm().item()

            print(
                f"{name}: "
                f"Grad Norm = {grad_norm:.8f}"
            )

    if not has_gradient:
        raise RuntimeError("所有模型参数均没有梯度")

# ============================================
# 11. Day20：单 Batch 过拟合测试
# ============================================

def overfit_one_batch():

    batch_x, batch_y = next(iter(train_loader))

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # 使用独立模型，避免影响正式训练
    test_model = FruitClassifier().to(device)

    test_optimizer = torch.optim.SGD(
        test_model.parameters(),
        lr=0.1
    )

    test_model.train()

    print("\n开始单 Batch 过拟合测试")

    # 使用普通 FP32 训练，减少干扰因素
    for step in range(200):

        test_optimizer.zero_grad()

        logits = test_model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        test_optimizer.step()

        if (step + 1) % 20 == 0:

            predictions = logits.argmax(dim=1)

            accuracy = (
                (predictions == batch_y)
                .float()
                .mean()
                .item()
            )

            print(
                f"Step {step + 1:3d} "
                f"Loss: {loss.item():.4f} "
                f"Accuracy: {accuracy:.2%}"
            )

# ============================================
# 12. Day20：验证集评估函数
# ============================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

# ============================================
# 13. 可选：单 Batch 过拟合测试
# ============================================

if single_batch_test:

    overfit_one_batch()

# ============================================
# 14. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_index, (batch_x, batch_y) in enumerate(
        train_loader
    ):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Day20：检查输入数据
        if debug_mode:
            check_batch(batch_x, batch_y)

        # 根据配置选择是否开启异常检测
        context = (
            torch.autograd.detect_anomaly()
            if detect_anomaly
            else nullcontext()
        )

        with context:

            # Day19：自动混合精度
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):

                logits = model(batch_x)

                # Day20：检查 Logits
                if debug_mode:

                    if not torch.isfinite(logits).all():
                        raise RuntimeError(
                            "Logits 存在 NaN 或 Inf"
                        )

                loss = criterion(logits, batch_y)

            # Day20：检查 Loss
            if debug_mode:

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Loss 存在 NaN 或 Inf"
                    )

            # =================================
            # Day17：梯度累积
            # =================================

            batch_size = batch_x.size(0)

            group_index = (
                batch_index // accumulation_steps
            )

            group_start = (
                group_index * effective_batch_size
            )

            group_size = min(
                effective_batch_size,
                len(train_dataset) - group_start
            )

            scaled_loss = (
                loss * batch_size / group_size
            )

            # =================================
            # Day20：人为制造梯度异常
            # =================================

            if fault_mode == "gradient_big":

                scaled_loss = scaled_loss * 10000

            elif fault_mode == "gradient_zero":

                scaled_loss = scaled_loss * 0.0

            # Day19：混合精度反向传播
            scaler.scale(scaled_loss).backward()

        # 判断是否完成一个累积周期
        is_full_cycle = (
            (batch_index + 1)
            % accumulation_steps == 0
        )

        is_last_batch = (
            batch_index + 1 == len(train_loader)
        )

        if is_full_cycle or is_last_batch:

            # Day19：还原梯度
            scaler.unscale_(optimizer)

            # Day20：检查每一层的真实梯度
            if debug_mode:

                check_gradients(
                    model,
                    print_details=(epoch == 0)
                )

            # Day18：梯度裁剪
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                error_if_nonfinite=debug_mode
            ).item()

            # Day20：梯度异常监控
            if debug_mode:

                if grad_norm > 100:

                    print(
                        f"警告：梯度范数过大 "
                        f"{grad_norm:.6f}"
                    )

                elif grad_norm < 1e-8:

                    print(
                        f"警告：梯度范数极小 "
                        f"{grad_norm:.10f}"
                    )

            # Day19：更新参数
            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad()

        # =====================================
        # 统计训练指标
        # =====================================

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_size

    # =========================================
    # 15. 计算训练指标
    # =========================================

    train_loss = total_loss / total_samples

    train_accuracy = (
        total_correct / total_samples
    )

    # Day20：计算验证集指标
    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}] "
        f"\nTrain Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2%} "
        f"\nVal Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2%}"
    )

print("\n训练完成！")
```

---

## 十、重点解释今天新增的代码

不再重复 Day 17–19 的内容，只讲今天新增的几个重要语法。

### 1\. `continue` 是什么？

```python
for name, param in model.named_parameters():

    if param.grad is None:
        continue

    print(param.grad.norm())
```

`continue` 表示跳过当前循环剩余的代码，直接进入下一次循环。

如果某个参数没有梯度，就不能对它执行 `.norm()`，所以直接跳过。

注意区分：

| 语法 | 作用 |
|--------|--------|
| `continue` | 跳过本次循环 |
| `break` | 结束整个循环 |
| `return` | 结束当前函数 |
| `raise` | 抛出异常 |

### 2\. `nullcontext()` 是什么？

```python
context = (
    torch.autograd.detect_anomaly()
    if detect_anomaly
    else nullcontext()
)

with context:
    ...
```

`nullcontext()` 是一个什么额外操作都不执行的上下文管理器。

这样写就能通过一个变量控制是否开启异常检测，而不需要写两套训练代码。

### 3\. `@torch.no_grad()` 是什么？

```python
@torch.no_grad()
def evaluate(model, loader):
    ...
```

它是一个装饰器，表示整个 `evaluate()` 函数都在不记录梯度的环境中运行。

等价于：

```python
def evaluate(model, loader):

    with torch.no_grad():
        ...
```

验证阶段不需要反向传播和参数更新，所以不需要构建用于反向传播的计算图。

同时注意：

```python
model.eval()
```

和：

```python
torch.no_grad()
```

是两个不同的机制。

前者切换模型的运行模式，影响 Dropout、BatchNorm 等层；后者关闭梯度记录。

---

## 十一、如何使用今天的代码？

今天不是简单地运行一次训练代码，而是通过修改 `fault_mode`，理解不同问题是如何被发现的。

> 原窗口场景选择器未返回选项说明。下面根据完整代码补充静态对照表。

| fault_mode | 人为制造的问题 | 预期现象 |
|---|---|---|
| `normal` | 无 | 完成训练和验证 |
| `nan_input` | 输入含 NaN | 全数据检查报错，训练尚未开始 |
| `bad_label` | 标签 5 超过三分类范围 | 全数据检查报错 |
| `gradient_big` | 累积损失乘 10000 | 监控到大梯度，之后执行裁剪；FP16 也可能先溢出并被 Debug 检查拦截 |
| `gradient_zero` | 累积损失乘 0 | 梯度为零并打印警告，参数无法学习 |

最后，将代码修改为：

```python
fault_mode = "normal"

single_batch_test = True
```

就可以观察单 Batch 过拟合测试的效果。

这里的异常场景是为了教学而人为构造的，梯度警告阈值 `100` 和 `1e-8` 也只是演示阈值，不是工业界的统一标准。代码未在实际 GPU 环境执行验证。

另外，Debug 模式采用发现异常梯度立即报错的策略；正常 FP16 训练中，则可以允许 GradScaler 跳过偶发的溢出更新。正式训练时应关闭开销较大的逐层检查。

---

## 十二、真正的工业级训练排查顺序

以后如果你在训练 GPT、Transformer 或其他大模型时遇到问题，可以按照下面的顺序排查。

> 原窗口交互清单未返回具体条目。以下按本课内容补充可在 GitHub 阅读的排查清单。

- [ ] 固定随机种子，记录异常配置，复现问题。
- [ ] 检查输入和标签的 Shape、dtype、范围与有限值。
- [ ] 检查 Logits，再检查 Loss，找到最早出现异常的位置。
- [ ] 在还原梯度之后检查每层梯度和总范数。
- [ ] 核对清零、累积、还原、裁剪与更新顺序。
- [ ] 用 FP32 单 Batch 过拟合测试缩小问题范围。
- [ ] 必要时启用异常检测，定位反向传播中的问题算子。
- [ ] 对照训练与验证指标判断过拟合；修复后重复验证。

---

## 十三、模块五总结（Day 17–20）

至此，整个训练优化与稳定性模块就结束了。

| 主题 | 解决的问题 | 核心代码 |
|--------|--------|--------|
| Day 17 梯度累积 | 显存不足，无法使用大 Batch | `scaled_loss.backward()` |
| Day 18 梯度裁剪 | 梯度过大 | `clip_grad_norm_()` |
| Day 19 混合精度 | 显存占用与计算效率 | `autocast`、`GradScaler` |
| Day 20 训练排查 | 定位训练异常 | `isfinite`、`detect_anomaly`、单 Batch 测试 |

现在，你的训练代码已经从最基础的：

```python
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

逐步发展为：

```python
with torch.autocast(...):
    logits = model(batch_x)
    loss = criterion(logits, batch_y)

# 检查 Loss
if not torch.isfinite(loss):
    raise RuntimeError("Loss 异常")

# 梯度累积 + 混合精度
scaled_loss = loss * batch_size / group_size

scaler.scale(scaled_loss).backward()

if is_full_cycle or is_last_batch:

    scaler.unscale_(optimizer)

    # 梯度裁剪
    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()
```

这已经是一个包含多种实际大模型训练机制的核心 Train Loop。

**Day 20 最重要的不是记住所有检测函数，而是形成工程排查思维：发现异常、缩小范围、定位原因、验证修复，而不是遇到问题就盲目调整**

> 上一句为接口返回的原文片段，末尾在“盲目调整”处被截断，未擅自补写原文。

---

## 归档与运行说明

来源：[《学习梯度累积》](https://chatgpt.com/c/6afac121-4a44-83e9-ac2f-5bac8be7fb03) 的 Day20 讲解。保存接口可取得的全部正文、例子、完整教学代码和语法解释，适配 GitHub 表格、流程图；未返回的交互内容和末尾截断均已注明。

`code.py` 为本地 `training_debug.py` 原样归档，当前 `fault_mode = "nan_input"`，因此直接运行会主动抛出“输入数据存在 NaN 或 Inf”，这是本次练习的预期结果。改为 `fault_mode = "normal"` 可完整训练；再设置 `single_batch_test = True` 可先观察单 Batch 过拟合。正文教学代码默认 `normal`，与本地练习的配置差异保留并在此说明。

原文“代码未在实际 GPU 环境执行验证”描述的是讲解当时的状态。本次归档另做实际运行验证，结果见 README。

补充边界：`param.grad is None` 与梯度张量全零不同；本课的 `gradient_zero` 制造的是后者。单 Batch 测试使用独立模型，不改变正式模型权重，但会消耗随机数状态，可能改变后续数据顺序。`assert` 用于本课教学检查，Python `-O` 会关闭断言；正常运行本练习无需 `-O`。梯度警告阈值只是演示，应结合训练阶段判断。
