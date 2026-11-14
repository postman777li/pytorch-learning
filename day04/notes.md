# Day 4：设备管理与模型模式

> 按原学习窗口的讲解顺序整理，保留设计动机、例子、反例、推导和逐步代码说明，并收录相关追问。原窗口交互组件转为静态表达，公式适配 GitHub。完整可运行的归档程序见 [code.py](code.py)。

好！我们正式进入 **PyTorch & Deep Learning Training — Day 4：设备管理与模型模式**。

今天是模块一的最后一天。Day 1–3 我们已经学会了 Tensor、自动求导和神经网络构建，今天要解决的是：**如何让模型在 CPU/GPU 上正确运行，以及如何控制模型在训练和推理时的行为。**

今天依然遵循你的学习要求：例子引入 → 设计动机 → 专业原理 → 代码实践 → 独立练习。尽量讲透，但不堆废话。



(模块一 · 最后一天) (约 60–90 分钟) (工程实践)

## 今天需要掌握的内容

| 知识点 | 核心问题 |
|--------|--------|
| CPU / GPU / CUDA | 为什么深度学习需要 GPU？ |
| `device` | 如何指定模型运行设备？ |
| `.to(device)` | 如何移动模型和数据？ |
| `train()` | 训练模式有什么特殊行为？ |
| `eval()` | 推理模式改变了什么？ |
| `no_grad()` | 为什么推理时通常关闭梯度？ |
| Parameters / Buffers | 模型中除了权重还有什么？ |
| 随机种子 | 如何让实验尽可能可复现？ |

---

# 第一部分：CPU、GPU 与 CUDA

## 1\. 具体例子引入

假设你要训练一个神经网络，需要计算：

```python
y = X @ W
```

其中：

- X：包含 10000 条样本。
- W：模型权重。
- `@`：矩阵乘法。

如果使用 CPU 计算，需要 CPU 完成大量数值运算。

如果使用 GPU，则可以利用 GPU 的大量并行计算能力，同时处理大量运算。


**CPU**

擅长复杂控制逻辑、通用计算和多种任务调度。


**GPU**

拥有大量计算单元，适合大规模并行数值运算。

### 2\. 为什么深度学习需要 GPU？

神经网络中有大量矩阵运算：

- Linear：矩阵乘法。
- Attention：Q、K、V 矩阵运算。
- FFN：多层线性变换。
- Backward：反向传播中的梯度计算。

这些计算有很强的并行性，因此 GPU 通常能显著加速模型训练。

### 3\. CUDA 是什么？

你可能经常看到：

```python
torch.cuda.is_available()
```

这里的 CUDA 是 NVIDIA 提供的 GPU 并行计算平台和编程生态。

可以这样理解：

- GPU：实际执行计算的硬件。
- CUDA：让程序能够使用 NVIDIA GPU 进行计算的软件平台。
- PyTorch：通过相关后端调用 GPU 完成 Tensor 运算。

注意：CUDA 并不是 GPU 本身，也不是所有 GPU 都支持 CUDA。PyTorch 还支持其他设备后端，例如 Apple Silicon 上的 MPS。

---

# 第二部分：device 与 .to(device)

(工程核心)

## 1\. 先看一个问题

假设我们创建了一个模型：

```python
model = MyModel()
```

默认情况下，模型参数通常在 CPU 上。

现在我们创建一个 GPU Tensor：

```python
x = torch.randn(2, 3, device="cuda")
```

然后：

```python
y = model(x)
```

如果模型还在 CPU，而输入 x 在 GPU，就会出现设备不一致错误。

**PyTorch 不允许普通 Tensor 在不同设备上直接进行大多数运算。**

所以我们需要统一设备。

## 2\. 如何判断 GPU 是否可用？

```python
import torch

print(torch.cuda.is_available())
```

可能输出：

```text
True
```

表示当前 PyTorch 环境可以使用 CUDA GPU。

如果输出：

```text
False
```

表示当前环境无法使用 CUDA，不一定代表电脑没有 GPU，也可能是驱动、PyTorch 安装版本等问题。

查看 GPU 数量：

```python
print(torch.cuda.device_count())
```

查看 GPU 名称：

```python
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 3\. device 是什么？

我们可以用一个变量统一管理设备：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

逐行理解：

```python
torch.cuda.is_available()
```

检查 CUDA 是否可用。

```python
A if condition else B
```

这是 Python 三元表达式：

- 条件成立，选择 A。
- 条件不成立，选择 B。

所以：

```python
device = torch.device("cuda")
```

或者：

```python
device = torch.device("cpu")
```

取决于当前环境。

## 4\. .to(device) 到底做什么？

假设：

```python
x = torch.randn(2, 3)
```

此时 x 默认在 CPU。

我们执行：

```python
x = x.to(device)
```

就是将 Tensor 移动到指定设备。

查看：

```python
print(x.device)
```

可能输出：

```text
cuda:0
```

其中 `cuda:0` 表示第 0 号 CUDA GPU。

### 模型也需要移动

```python
model = MyModel()

model = model.to(device)
```

这会将模型中已注册的参数和缓冲区移动到目标设备。

### 最重要的工程规则

**参与同一次运算的模型、输入数据和目标标签，通常需要放在兼容的设备上。**

例如：

```python
model = MyModel().to(device)

X = X.to(device)
y = y.to(device)

pred = model(X)
loss = loss_fn(pred, y)
```

不要只移动模型，却忘记移动输入数据。

---

# 第三部分：train() 与 eval()

这是今天另一个非常重要的知识点。

## 1\. 具体例子引入

假设我们已经训练好了一个模型。

现在有两种情况：

**情况 A：模型正在训练。**

我们希望模型不断调整参数，学习数据中的规律。

**情况 B：模型已经训练完成，现在只需要预测。**

我们希望模型稳定地输出预测结果，不再进行训练阶段特有的行为。

这两种场景需要不同的模型模式。

PyTorch 提供：

```python
model.train()
model.eval()
```

## 2\. train()：训练模式

```python
model.train()
```

告诉模型：现在进入训练模式。

## 3\. eval()：评估模式

```python
model.eval()
```

告诉模型：现在进入评估/推理模式。

但这里有一个关键问题：

**这两个方法到底改变了什么？**

它们主要影响 Dropout、BatchNorm 等具有训练和推理差异的模块。

### 例子一：Dropout

假设一个神经网络有 100 个神经元。

训练时，我们希望模型不要过度依赖某几个神经元，因此随机屏蔽一部分神经元。

这就是 Dropout 的作用之一。

```python
dropout = nn.Dropout(p=0.5)
```

其中：

- `p=0.5`：训练时随机将约 50% 的元素置零。
- 剩余元素会进行缩放，以保持期望值不变。

看代码：

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)

x = torch.ones(10)

# 训练模式
dropout.train()
print(dropout(x))

# 推理模式
dropout.eval()
print(dropout(x))
```

训练模式下，输出大致类似：

```text
tensor([2., 0., 2., 0., 0., 2., 2., 0., 2., 0.])
```

每次运行结果可能不同。

为什么有些位置变成 0，有些变成 2？

- 随机屏蔽的元素变成 0。
- 保留下来的元素除以 `1-p`，所以这里变成 2。

推理模式下：

```text
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

不再随机屏蔽神经元，也不再缩放。

**总结：Dropout 在训练时随机屏蔽部分元素，推理时关闭 Dropout。**

### 例子二：BatchNorm

BatchNorm 是一种归一化方法。

它会根据一批数据的均值和方差对数据进行归一化。

训练模式：

```python
model.train()
```

BatchNorm 使用当前 Batch 的均值和方差进行归一化，并更新运行均值、运行方差等统计信息。

推理模式：

```python
model.eval()
```

BatchNorm 使用训练期间积累的运行均值和运行方差，不再更新这些统计信息。

这里不要求你今天深入学习 BatchNorm 的数学公式，先掌握它为什么需要区分训练和推理模式。

## 4\. train() 和 eval() 的重要区别

| 模块 | `train()` | `eval()` |
|--------|--------|--------|
| Dropout | 随机屏蔽 | 不再屏蔽 |
| BatchNorm | 使用当前 Batch 统计量并更新运行统计量 | 使用运行统计量，不更新 |
| 普通 Linear | 正常计算 | 正常计算 |

注意：

**`train()` 和 `eval()` 不会自动开启或关闭梯度计算！**

这是非常容易混淆的地方。

---

# 第四部分：eval() 和 no\_grad() 有什么区别？

这部分一定要搞清楚。

你 Day 2 已经学过：

```python
with torch.no_grad():
```

它和 `model.eval()` 完全不是一回事。

### 1\. model.eval()

```python
model.eval()
```

作用：

**切换模型内部模块的行为模式。**

例如关闭 Dropout，固定 BatchNorm 的运行统计量。

### 2\. torch.no\_grad()

```python
with torch.no_grad():
    output = model(x)
```

作用：

**不记录这段普通运算的自动求导历史。**

这样就不需要为这段计算建立通常的反向传播计算图，可以减少内存开销。

### 3\. 两者可以同时使用

实际推理通常这样写：

```python
model.eval()

with torch.no_grad():
    output = model(x)
```

为什么两个都需要？

- `eval()`：告诉模型使用推理模式。
- `no_grad()`：告诉 PyTorch 不需要记录梯度。

**训练阶段**

```python
model.train()

pred = model(x)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

**推理阶段**

```python
model.eval()

with torch.no_grad():
    pred = model(x)
```

记住这个区别：

| 对比 | `model.eval()` | `torch.no_grad()` |
|--------|--------|--------|
| 控制什么 | 模型运行模式 | 梯度记录 |
| 影响 Dropout | 是 | 否 |
| 影响 BatchNorm | 是 | 否 |
| 关闭梯度记录 | 否 | 是 |
| 能否同时使用 | 可以 | 可以 |

特别提醒：即使执行了 `model.eval()`，只要没有关闭梯度记录，模型仍然可以计算梯度。

---

# 第五部分：模型参数与缓冲区 Parameters & Buffers

这是今天稍微进阶一点的内容，但它对理解模型内部结构很重要。

## 1\. 先看一个问题

我们之前学过：

```python
self.linear = nn.Linear(2, 1)
```

这个 Linear 里面有什么？

```python
linear.weight
linear.bias
```

它们是模型的可训练参数。

但模型内部并不只有需要训练的权重。

例如 BatchNorm 还有：

- 运行均值 `running_mean`
- 运行方差 `running_var`

这些数值需要保存，但不需要通过梯度下降进行训练。

这就需要另一种机制：**Buffer（缓冲区）。**

## 2\. Parameter 和 Buffer 的区别

| 对比 | Parameter | Buffer |
|--------|--------|--------|
| 是否是模型状态 | 是 | 是 |
| 是否参与梯度训练 | 通常是 | 通常否 |
| 是否出现在 `parameters()` | 是 | 否 |
| 是否默认进入 `state_dict()` | 是 | 是（持久化 Buffer） |
| 是否随 `model.to(device)` 移动 | 是 | 是 |

## 3\. 如何注册 Buffer？

PyTorch 提供：

```python
self.register_buffer()
```

看代码：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor([2.0])
        )

        self.register_buffer(
            "running_value",
            torch.tensor([10.0])
        )

    def forward(self, x):
        return x * self.weight + self.running_value
```

这里：

```python
self.weight = nn.Parameter(...)
```

表示创建一个可训练参数。

而：

```python
self.register_buffer(
    "running_value",
    torch.tensor([10.0])
)
```

表示注册一个缓冲区。

为什么需要注册？

因为注册之后，PyTorch 才能统一管理它：

- 跟随模型移动设备。
- 默认保存到 `state_dict()`。
- 不会被当成普通可训练参数交给优化器。

### 4\. 代码验证

```python
model = MyModel()

print(list(model.named_parameters()))
print(list(model.named_buffers()))
print(model.state_dict())
```

你会发现：

- `weight` 出现在 `named_parameters()` 中。
- `running_value` 出现在 `named_buffers()` 中。
- 两者都会出现在默认的 `state_dict()` 中。

如果不希望某个 Buffer 被保存，可以使用：

```python
self.register_buffer(
    "temporary_value",
    torch.tensor([1.0]),
    persistent=False
)
```

这样它仍然会跟随模型移动设备，但不会进入 `state_dict()`。

**记忆：Parameter 是模型可训练的参数；Buffer 是模型需要管理和保存、但不作为普通可训练参数的状态。**

---

# 第六部分：随机种子与可复现性

## 1\. 具体例子引入

假设你训练了一个模型：

```python
model = nn.Linear(2, 1)

print(model.weight)
```

然后你重新运行：

```python
model = nn.Linear(2, 1)

print(model.weight)
```

你会发现两次权重可能不一样。

为什么？

因为 PyTorch 初始化模型参数时，通常会使用随机数。

同样：

```python
x = torch.rand(2, 3)
```

每次生成的结果也可能不同。

这会带来一个问题：

如果两次实验的初始参数、随机数据都不同，我们很难判断训练结果的差异究竟来自哪里。

所以需要随机种子。

## 2\. torch.manual\_seed()

```python
import torch

torch.manual_seed(42)

a = torch.rand(3)

print(a)
```

重新运行：

```python
torch.manual_seed(42)

b = torch.rand(3)

print(b)
```

在相同环境、相同随机数调用顺序下，通常会得到相同结果。

这里的：

```python
42
```

只是一个随机种子，可以换成其他整数。

### 3\. GPU 随机种子

如果使用 CUDA，还可以设置：

```python
torch.cuda.manual_seed_all(42)
```

它会为所有 CUDA 设备设置随机种子。

### 4\. 更完整的可复现性设置

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

分别对应：

| 代码 | 控制对象 |
|--------|--------|
| `random.seed()` | Python 随机数 |
| `np.random.seed()` | NumPy 随机数 |
| `torch.manual_seed()` | PyTorch 随机数 |
| `torch.cuda.manual_seed_all()` | 所有 CUDA 设备的随机数 |

### 5\. 设置随机种子就一定能完全复现吗？

不一定。

某些 GPU 算子可能存在非确定性行为，另外不同硬件、CUDA、PyTorch 版本也可能导致结果差异。

如果确实需要尽可能严格的确定性实验，还可以设置：

```python
torch.use_deterministic_algorithms(True)
```

但这可能导致某些运算变慢，或者在没有确定性实现的操作上报错。

今天先记住：

**随机种子能够帮助控制随机初始化和随机数生成，但不保证所有环境下完全一致。**

---

# 第七部分：完整代码实践 device\_and\_mode.py

现在把今天的知识组合起来。

我们构建一个简单神经网络，完成：

1. 自动选择 CPU/GPU。
2. 将模型和数据移动到对应设备。
3. 进入训练模式。
4. 训练模型。
5. 进入评估模式。
6. 关闭梯度记录进行预测。
7. 查看模型参数和 Buffer。
8. 设置随机种子。

## 完整代码

```python

import torch
import torch.nn as nn
import random
import numpy as np

# ==================================
# 1. 设置随机种子
# ==================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ==================================
# 2. 选择设备
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

# ==================================
# 3. 定义模型
# ==================================

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ==================================
# 4. 创建模型并移动设备
# ==================================

model = MyModel().to(device)

print("模型设备:", next(model.parameters()).device)

# ==================================
# 5. 创建训练数据并移动设备
# ==================================

X = torch.rand(200, 2)
y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1

X = X.to(device)
y = y.to(device)

# ==================================
# 6. 定义损失函数和优化器
# ==================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ==================================
# 7. 训练模型
# ==================================

model.train()

for epoch in range(200):

    pred = model(X)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )

# ==================================
# 8. 评估模型
# ==================================

model.eval()

test_x = torch.tensor(
    [[0.5, 0.8]]
).to(device)

with torch.no_grad():

    test_pred = model(test_x)

print("预测结果:", test_pred.item())
print("真实结果:", 4.4)

# ==================================
# 9. 查看模型参数和缓冲区
# ==================================

print("\n模型参数:")

for name, param in model.named_parameters():
    print(name, param.shape, param.device)

print("\n模型缓冲区:")

for name, buffer in model.named_buffers():
    print(name, buffer.shape, buffer.device)

# ==================================
# 10. 保存模型权重
# ==================================

torch.save(
    model.state_dict(),
    "model_weights.pth"
)

print("\n模型权重保存完成")
```

## 运行逻辑

注意：这段代码使用了 Dropout，因此训练和推理时的输出行为不同。训练损失可能有一定波动，不必要求每次都完全相同。

---

# 第八部分：Day 4 独立练习

今天的练习主要检查你是否真正掌握设备管理和模型模式。

原讲解在此提供交互练习，下文保留可读取的讲解与验收目标。
原窗口的交互练习未提供可读取的题目和作答，按下方知识点自行核对。


---

# Day 4 最终总结

今天最重要的是记住下面这张表：

| API | 作用 |
|--------|--------|
| `torch.cuda.is_available()` | 检查 CUDA 是否可用 |
| `torch.device()` | 指定运行设备 |
| `.to(device)` | 移动模型、Tensor |
| `model.train()` | 切换训练模式 |
| `model.eval()` | 切换评估模式 |
| `torch.no_grad()` | 关闭梯度记录 |
| `register_buffer()` | 注册非普通可训练状态 |
| `torch.manual_seed()` | 设置 PyTorch 随机种子 |
| `state_dict()` | 获取模型状态 |

尤其要区分：

- `train()` / `eval()`：控制模型行为。
- `no_grad()`：控制是否记录梯度。
- `.to(device)`：控制数据和模型在哪个设备上运行。

**到这里，PyTorch 核心基础 Day 1–4 的知识就全部学完了。**

完成今天的练习后，我们就可以进入下一模块：模型训练流程，重点学习 Dataset、DataLoader、Batch、Loss、Optimizer，以及真正接近工业训练的完整 Pipeline。

---

## 与本地练习对应的工程说明

[code.py](code.py) 来自 device_and_mode.py。整理时修正两处 is_available 漏括号，权重保存到脚本旁。判断要写 torch.cuda.is_available()，函数对象本身在条件中会被视为真值。

Tensor 迁移使用 x=x.to(device) 接收返回结果。Module.to(device) 原地迁移注册参数及 Buffer，并返回该 Module。建议先迁移模型，再创建优化器。

第一段 MyModel 注册 weights 与 running_value，但 forward 只计算 self.net(x)，因此该 weights 不参与 Loss，梯度为 None。后面重定义的训练模型只有 Linear、ReLU、Dropout，named_buffers() 返回空列表符合其结构。

BatchNorm 默认 track_running_stats=True。若设为 False，评估也使用当前批次统计量。固定随机种子要结合相同调用顺序与环境，跨设备结果可能不同。
