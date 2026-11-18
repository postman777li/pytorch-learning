# Day 19：Mixed Precision（混合精度训练）

(模块五 · 训练优化与稳定性) (Day 19 / 20)

今天我们进入一个非常重要的大模型训练技术：**混合精度训练**。

之前 Day 17 解决了显存不足（梯度累积），Day 18 解决了梯度过大（梯度裁剪）。

今天解决另一个问题：

**能不能让模型在保证训练效果的同时，减少显存占用、提高训练速度？**

今天重点掌握：

1. FP32、FP16、BF16 到底是什么？
2. 为什么低精度能够节省显存、提高速度？
3. `autocast` 是什么？
4. `GradScaler` 是什么？为什么 FP16 需要它？
5. 如何在之前的 FruitClassifier 训练代码中加入混合精度？

---

## 一、具体例子：为什么计算一定要使用 FP32？

假设你要计算：

$
1.23456789\times 2.34567891
$

计算机需要用一定数量的二进制位来存储这些数字。

如果使用更多位数，通常能够表示更精确的数值；如果使用更少位数，存储和计算的资源需求可能更低，但精度也会降低。

这就像：

- FP32：使用更精细的刻度尺。
- FP16：使用刻度更粗的尺子。
- BF16：也是较粗的尺子，但它能够覆盖更大的数值范围。

那么问题来了：

**训练神经网络时，每一次计算都必须使用最精确的 FP32 吗？**

其实不需要。

很多神经网络运算即使使用较低精度，也能得到足够好的结果。

例如，矩阵乘法、线性层等计算，在支持相应硬件的 GPU 上，使用 FP16 或 BF16 往往更高效。

所以我们可以：

- 对一些计算使用低精度。
- 对一些需要更高数值稳定性的计算保留 FP32。

这就是混合精度训练的基本思想。

---

## 二、专业概念：FP32、FP16、BF16

首先认识三个概念。

### 1\. 它们分别是什么？

| 类型 | 每个数值占用 | 含义 |
|--------|--------|--------|
| FP32 | 32 bit（4字节） | 单精度浮点数 |
| FP16 | 16 bit（2字节） | 半精度浮点数 |
| BF16 | 16 bit（2字节） | Brain Floating Point 16 |

它们最大的区别在于：**分配给数值范围和精度的二进制位数不同。**

**浮点数的二进制位分配**

| 类型 | 符号位 | 指数位 | 尾数位 | 总位数 |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 32 |
| FP16 | 1 | 5 | 10 | 16 |
| BF16 | 1 | 8 | 7 | 16 |

不用死记二进制位数，但必须理解两个概念：

- **指数位**：主要决定能够表示的数值范围。
- **尾数位**：主要决定数值的精细程度。

### 2\. FP16 和 BF16 有什么区别？

这是今天非常重要的知识点。

| 对比 | FP16 | BF16 |
|--------|--------|--------|
| 总位数 | 16 | 16 |
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 数值范围 | 较小 | 接近 FP32 |
| 精度 | 相对更高 | 相对更低 |
| 数值稳定性 | 更容易溢出或下溢 | 通常更稳定 |

这里重点理解：

**BF16 虽然精度比 FP16 低，但是数值范围更大。**

为什么？

因为 BF16 的指数位有 8 位，和 FP32 一样。

这使它能够表示非常大和非常小的数值，不容易出现 FP16 那种数值溢出、下溢的问题。

所以在现代大模型训练中，BF16 非常常见。

### 3\. 为什么 FP16 容易出现数值问题？

假设训练过程中产生一个非常小的梯度：

$
g=0.00000001
$

FP16 能表示的最小正数范围有限。

如果某个数值太小，可能在转换为 FP16 时变成 0。

这叫做：

**Underflow（下溢）**。

如果数值太大，超过 FP16 的表示范围，则可能变成无穷大 Inf。

这叫做：

**Overflow（溢出）**。

这两种情况都可能影响模型训练。

BF16 因为数值范围更大，通常能更好地避免这类问题。

---

## 三、为什么混合精度能够节省显存、提高速度？

### 1\. 显存占用

假设有一个张量：

```python
x = torch.randn(1000, 1000)
```

它包含 100 万个元素。

如果使用 FP32：

$
1000000\times4=4MB
$

如果使用 FP16：

$
1000000\times2=2MB
$

**同样 100 万个元素**

**FP32**

4 MB

**FP16**

2 MB

(该张量存储空间减少 50%)

注意：这里说的是同一个张量的数据存储空间减少 50%，不代表整个训练过程的显存一定减少 50%。

因为模型参数、优化器状态、梯度、激活值等都会占用显存。

### 2\. 为什么计算速度可能更快？

现代 NVIDIA GPU 提供了专门针对低精度矩阵计算的硬件单元，例如 Tensor Cores。

这些硬件可以高效执行 FP16、BF16 等低精度计算。

因此，混合精度训练往往能够提高矩阵乘法等运算的吞吐量。

当然，具体提升多少取决于 GPU 型号、模型结构和算子支持情况。

---

## 四、autocast：自动选择计算精度

现在进入 PyTorch 代码。

以前我们的训练代码是：

```python
logits = model(batch_x)

loss = criterion(logits, batch_y)

loss.backward()
```

现在我们可以使用：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
```

这就是自动混合精度（Automatic Mixed Precision，AMP）。

### 1\. `autocast` 到底做了什么？

你可以把它理解为一个自动精度管理器。

它会根据不同运算的特点，自动选择合适的计算精度。

例如：

| 运算 | 精度处理 |
|--------|--------|
| 某些矩阵乘法 | 可以使用 FP16/BF16 |
| 某些数值敏感运算 | 保持 FP32 |
| 某些损失计算 | 通常使用 FP32 |

所以：

**autocast 并不是把整个模型强制转换成 FP16，而是根据运算类型自动选择精度。**

这也是为什么我们叫它混合精度。

### 2\. 为什么要使用 `with`？

你之前学过 Python 的上下文管理器。

```python
with torch.autocast(...):
    logits = model(batch_x)
```

意思是：

在这个代码块执行期间，启用自动混合精度。

离开代码块后，恢复原来的精度管理状态。

特别注意：

通常不需要再手动写：

```python
model.half()
```

因为 autocast 会自动处理适合低精度执行的运算。

---

## 五、GradScaler：为什么 FP16 需要它？

现在来到今天第二个核心 API。

我们刚才说过，FP16 容易出现梯度下溢。

那么有没有办法解决？

有。

这就是 GradScaler（梯度缩放器）。

### 1\. 先通过一个例子理解

假设模型计算得到的梯度是：

$
g=0.000001
$

如果 FP16 表示过程中出现精度问题，梯度可能变得非常小，甚至直接变成 0。

这会导致什么？

模型本来应该更新参数，但是梯度变成了 0，参数就可能无法正常更新。

GradScaler 的思路是：

**在反向传播之前，先把 Loss 放大。**

假设放大 10000 倍：

$
Loss'=Loss\times10000
$

那么反向传播得到的梯度也会放大 10000 倍：

$
g'=g\times10000
$

原本：

$
g=0.000001
$

现在：

$
g'=0.01
$

这样就能降低小梯度下溢的风险。

### 2\. 那参数更新怎么办？

梯度被放大了 10000 倍，不能直接拿它更新参数。

因此在更新之前，还需要把梯度缩小回来：

$
g=\frac{g'}{10000}
$

GradScaler 会自动处理这个过程。

### 3\. PyTorch 怎么使用？

```python
scaler = torch.amp.GradScaler("cuda")

# 放大 Loss 后进行反向传播
scaler.scale(loss).backward()

# 更新参数
scaler.step(optimizer)

# 动态调整缩放倍数
scaler.update()
```

这三个 API 必须理解：

| API | 作用 |
|--------|--------|
| `scaler.scale(loss)` | 放大 Loss |
| `scaler.step(optimizer)` | 还原梯度并执行优化器更新（内部先检查梯度是否异常） |
| `scaler.update()` | 动态调整缩放倍数 |

如果检测到梯度出现 Inf 或 NaN，GradScaler 通常会跳过这次参数更新，并调整缩放因子。

### 4\. 为什么 BF16 通常不需要 GradScaler？

因为 BF16 的指数位与 FP32 一样，能够表示的数值范围非常大。

因此，梯度下溢和溢出的风险相对更低。

所以：

| 精度 | 通常是否需要 GradScaler |
|--------|--------|
| FP32 | 不需要 |
| FP16 | 通常需要 |
| BF16 | 通常不需要 |

记住：GradScaler 主要解决 FP16 训练中的数值范围问题，并不是所有混合精度训练都必须使用它。

---

## 六、代码实践：将混合精度加入 Day 17 的训练循环

现在我们把前面学到的知识结合起来。

今天的完整训练流程是：

1. Forward：在 `autocast` 环境下执行前向传播
2. Loss：计算损失
3. Loss Scaling：通过 `scaler.scale()` 放大 Loss
4. Backward：计算并累积梯度
5. Unscale：将累积梯度还原为真实尺度
6. Gradient Clipping：执行梯度裁剪
7. Optimizer Step：更新参数
8. Scaler Update：动态调整缩放倍数

注意，Day 17 的梯度累积还在，Day 18 的梯度裁剪也还在。

因此，今天我们需要特别注意 GradScaler 与这两个机制的配合。

### 一个非常重要的顺序

```python
scaler.scale(scaled_loss).backward()

# 累积周期结束后：

scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

scaler.step(optimizer)
scaler.update()

optimizer.zero_grad()
```

为什么要先 `unscale_()` 再裁剪？

因为 `scaler.scale()` 放大了梯度。

如果直接裁剪，裁剪的就是放大后的梯度，结果就不正确了。

所以顺序必须是：

**梯度缩放 → 反向传播 → 梯度还原 → 梯度裁剪 → 参数更新。**

---

## 七、Day 19 完整代码：mixed\_precision.py

这次继续使用你的 FruitClassifier，不改变模型结构。

代码还会自动判断 GPU 是否支持 BF16：

- 支持 BF16：优先使用 BF16。
- 不支持 BF16：使用 FP16 + GradScaler。
- 没有 CUDA：退回 FP32，保证代码仍然可以运行。

```python

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

# ============================================
# 2. 选择设备和混合精度类型
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 判断是否使用 CUDA
use_amp = device.type == "cuda"

# 如果 GPU 支持 BF16，优先使用 BF16
if use_amp and torch.cuda.is_bf16_supported():

    amp_dtype = torch.bfloat16
    precision_name = "BF16"

else:

    amp_dtype = torch.float16
    precision_name = "FP16"

# GradScaler 只在 FP16 CUDA 训练时启用
use_scaler = (
    use_amp and amp_dtype == torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_scaler
)

print("当前设备:", device)

if use_amp:
    print("当前混合精度:", precision_name)
else:
    print("当前使用 FP32")

print("GradScaler 是否启用:", use_scaler)

# ============================================
# 3. 创建合成训练数据
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

print("X shape:", x.shape)
print("y shape:", y.shape)

# ============================================
# 4. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ============================================
# 5. 定义模型
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

# ============================================
# 6. 模型、损失函数、优化器
# ============================================

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# ============================================
# 7. 训练参数
# ============================================

num_epochs = 10

# Day17：梯度累积
accumulation_steps = 4

micro_batch_size = loader.batch_size

effective_batch_size = (
    micro_batch_size * accumulation_steps
)

# Day18：梯度裁剪
max_norm = 1.0

# ============================================
# 8. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_index, (batch_x, batch_y) in enumerate(loader):

        # 将数据移动到 GPU 或 CPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # ====================================
        # Day19：autocast 自动混合精度
        # ====================================

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):

            # 1. Forward
            logits = model(batch_x)

            # 2. Calculate Loss
            loss = criterion(logits, batch_y)

        # ====================================
        # Day17：梯度累积
        # ====================================

        batch_size = batch_x.size(0)

        group_index = batch_index // accumulation_steps

        group_start = group_index * effective_batch_size

        group_size = min(
            effective_batch_size,
            len(dataset) - group_start
        )

        scaled_loss = loss * batch_size / group_size

        # ====================================
        # Day19：GradScaler
        # ====================================

        # FP16 时放大 Loss
        # BF16 / FP32 时自动退化为普通 backward
        scaler.scale(scaled_loss).backward()

        # 判断是否更新参数
        is_full_cycle = (
            (batch_index + 1) % accumulation_steps == 0
        )

        is_last_batch = (
            (batch_index + 1) == len(loader)
        )

        if is_full_cycle or is_last_batch:

            # =================================
            # Day19：还原梯度
            # =================================

            scaler.unscale_(optimizer)

            # =================================
            # Day18：梯度裁剪
            # =================================

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm
            )

            # =================================
            # Day19：更新参数
            # =================================

            scaler.step(optimizer)

            # 更新 GradScaler 的缩放因子
            scaler.update()

            # 清零梯度
            optimizer.zero_grad()

            if epoch == 0:
                print(
                    f"Batch {batch_index + 1} "
                    f"Grad Norm: {grad_norm.item():.4f}"
                )

        # ====================================
        # 9. 统计训练指标
        # ====================================

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        correct = (predictions == batch_y).sum().item()

        total_correct += correct

        total_samples += batch_size

    # ========================================
    # 10. 计算 Epoch 指标
    # ========================================

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )

# ============================================
# 11. 训练结束
# ============================================

print("\n训练完成！")
```

---

## 八、今天最重要的代码细节

### 1\. 为什么模型要 `.to(device)`？

```python
model = FruitClassifier().to(device)
```

因为我们需要让模型参数和输入数据处于同一个设备。

例如：

```python
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)
```

如果模型在 GPU，数据却在 CPU，就会发生设备不匹配错误。

### 2\. 为什么 `autocast` 只包住 Forward 和 Loss？

```python
with torch.autocast(...):
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
```

因为这部分主要是模型前向计算和损失计算，autocast 会根据算子的数值特性选择精度。

后面的梯度缩放、梯度累积和优化器更新由对应机制管理，不需要再放进 autocast 代码块。

### 3\. 为什么 `scaler.unscale_()` 必须放在裁剪前？

假设原始梯度范数为：

$
\|g\|=2
$

GradScaler 将梯度放大 1000 倍：

$
\|g'\|=2000
$

如果我们直接裁剪到 1：

$
g''=g'\times\frac{1}{2000}
$

然后再还原梯度，就会得到一个被错误缩小的梯度。

因此必须：

```python
scaler.unscale_(optimizer)

nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

先还原，再裁剪。

### 4\. 为什么 `scaler.step()` 取代了 `optimizer.step()`？

以前：

```python
optimizer.step()
```

现在：

```python
scaler.step(optimizer)
```

因为 GradScaler 需要先检查梯度是否存在 Inf 或 NaN。

如果梯度正常，就执行优化器更新；如果梯度异常，就跳过本次更新。

---

## 九、Day 19 知识总结

| 知识点 | 必须掌握 |
|--------|--------|
| FP32 | 32 位单精度 |
| FP16 | 16 位半精度，数值范围较小 |
| BF16 | 16 位，数值范围接近 FP32 |
| Mixed Precision | 不同计算使用不同精度 |
| autocast | 自动选择运算精度 |
| GradScaler | 主要解决 FP16 梯度下溢问题 |
| scale() | 放大 Loss |
| unscale\_() | 还原梯度 |
| step() | 检查梯度并更新参数 |
| update() | 动态调整缩放因子 |

**今天最重要的三组代码：**

```python
# 1. 自动混合精度
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16
):
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
```

```python
# 2. 梯度缩放与反向传播
scaler.scale(loss).backward()
```

```python
# 3. 还原梯度、裁剪、更新
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(), 1.0
)

scaler.step(optimizer)
scaler.update()
```

---

## 十、Day 19 自测

今天不需要做额外作业，但我希望你先回答下面三个问题，检查是否真正理解。

> 原窗口的交互式自测没有返回题目内容，因此这里只保留可取得的自测说明，不补写原题。

今天的学习重点是：**理解 FP16 和 BF16 的区别，真正掌握 autocast 与 GradScaler 的配合，并且能独立把它们加入训练循环。**

完成 Day 19 后，我们就剩最后一天：Day 20 Training Debug &amp; Stability，学习如何真正定位训练过程中的各种异常。

---

## 归档说明

来源：[《学习梯度累积》](https://chatgpt.com/c/6afac121-4a44-83e9-ac2f-5bac8be7fb03) 的 Day19 完整讲解。保留讲解顺序、设计动机、数值例子、公式、API 解释与完整教学代码，将位分配示意转为表格、公式适配 GitHub。`code.py` 为本地 `mixed_precision.py` 原样归档。

补充理解：`scaled_loss` 是梯度累积按样本数加权后的损失；`scaler.scale()` 是 FP16 的数值缩放，两者目的不同。一个累积周期内保持相同的 GradScaler 缩放因子，仅在周期末调用一次 `unscale_()`、裁剪、`step()` 和 `update()`。本例的 `group_size` 写法适用于当前固定 Batch Size、`drop_last=False` 的 DataLoader。
