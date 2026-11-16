# Day 11：完整训练循环（Train Loop）

> 来源为《学习Day9前向与损失》，保留原讲解顺序、设计动机、例子、反例、推导与逐步代码说明，交互图表转为静态表格，公式适配 GitHub。归档程序见 [code.py](code.py)。

(Day 11 / 12) (模块三：模型训练完整流程)

今天是模块三非常关键的一天。

在 Day 9 和 Day 10 中，你已经掌握：

```python
logits = model(x)                 # Forward
loss = criterion(logits, labels)  # Loss

optimizer.zero_grad()             # 清零梯度
loss.backward()                   # 反向传播
optimizer.step()                  # 更新参数
```

但是，之前的代码只完成了**一次参数更新**。

真实的模型训练需要不断读取数据、重复更新参数，直到完成整个训练过程。

**今天的目标：把你学过的 Dataset、DataLoader、Forward、Loss、Backward、Optimizer 全部串起来，独立写出真正的 PyTorch 训练循环。**

按照你的学习习惯，仍然是：具体例子 → 设计动机 → 专业概念 → 逐步代码 → 完整代码。

---

## 第一步：用一个具体例子理解训练循环

假设我们要训练一个水果分类模型。

现在有 100 条水果数据，每条数据包含两个特征：

- 甜度
- 重量

模型需要判断它属于苹果、香蕉还是橙子。

如果有 100 条数据，难道要手动写 100 次下面的代码吗？

```python
logits = model(x)
loss = criterion(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

当然不用。

我们可以把数据分成多个 Batch，然后让程序自动循环。

假设：

- 数据量：100 条
- Batch Size：20
- Epoch：3

那么整个训练过程如下。

训练数据

100 条数据，分成 5 个 Batch，每个 Batch 20 条

---

最终：

3 个 Epoch × 每个 Epoch 5 个 Batch = **15 次参数更新**

假设每个 Batch 更新一次，不使用梯度累积。

这就是训练循环的核心：

**外层循环控制训练多少轮，内层循环控制每轮处理多少个 Batch。**

---

## 第二步：为什么要这样设计？

### 2.1 为什么不一次性把全部数据输入模型？

假设你有 100 万条训练数据。

如果一次全部送进 GPU，可能直接显存不足（Out of Memory）。

因此，我们把数据拆成多个 Batch。

每次只处理一部分：

```python
Batch 1 → 训练 → 更新参数
Batch 2 → 训练 → 更新参数
Batch 3 → 训练 → 更新参数
...
```

这样可以控制显存使用，并利用批量并行计算提高效率。

注意：Batch Size 不是越小越好，也不是越大越好，它涉及显存、计算效率和训练效果之间的权衡。

### 2.2 为什么需要多个 Epoch？

模型只看一遍数据，通常还没有充分学习数据中的规律。

所以需要反复训练。

例如：

```text
Epoch 1：模型刚开始学习

Epoch 2：继续调整参数

Epoch 3：进一步优化预测

Epoch 4：继续训练
```

但也不是 Epoch 越多越好。

训练太久可能导致过拟合，这就是 Day 12 要学习的内容。

### 2.3 为什么要记录训练过程中的指标？

如果我们只让模型不断训练：

```python
for epoch in range(10):
    ...
```

却不打印任何信息，就无法知道：

- 模型的 Loss 是否下降。
- 模型的 Accuracy 是否提高。
- 训练是否正常进行。
- 是否出现数值异常。

因此，真实训练代码通常还需要记录 Loss、Accuracy 等指标。

---

# 第三步：专业概念与原理

## 3.1 Batch、Iteration、Epoch

这三个概念一定要分清楚。

| 概念 | 定义 | 举例 |
|--------|--------|--------|
| Batch | 一次送入模型的一组样本 | 20 条水果数据 |
| Batch Size | 一个 Batch 中的样本数量 | 20 |
| Iteration | 训练循环的一次迭代，通常处理一个 Batch | 处理 Batch 1 |
| Epoch | 完整遍历一遍训练数据 | 处理全部 100 条数据 |

假设：

- Dataset 有 105 条数据。
- Batch Size = 20。
- Epoch = 3。

一个 Epoch 需要几个 Batch？

$$
\left\lceil\frac{105}{20}\right\rceil=6
$$

因为：

```text
Batch 1：20 条
Batch 2：20 条
Batch 3：20 条
Batch 4：20 条
Batch 5：20 条
Batch 6： 5 条
```

最后一个 Batch 只有 5 条数据。

因此：

$$
\text{每轮 Batch 数}=6
$$

三个 Epoch：

$$
3\times6=18
$$

若每个 Batch 更新一次，就是 18 次参数更新。

PyTorch 中，可以直接查看：

```python
len(dataset)  # 样本总数量

len(loader)   # 一个 Epoch 中的 Batch 数
```

需要注意，如果 `DataLoader(drop_last=True)`，那么最后不足一个 Batch 的数据会被丢弃。

对于普通的固定大小 Dataset，默认 `drop_last=False`。

---

## 3.2 Train Loop：训练循环

最核心的代码结构：

```python
for epoch in range(num_epochs):

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()
```

我们逐行理解。

### 外层循环

```python
for epoch in range(num_epochs):
```

假设：

```python
num_epochs = 3
```

那么：

```python
range(3)
```

依次产生：

```python
0, 1, 2
```

所以 `epoch` 会依次等于 0、1、2。

如果你想打印成第 1、2、3 轮：

```python
print(epoch + 1)
```

### 内层循环

```python
for batch_x, batch_y in loader:
```

你之前 Day 6 学过 DataLoader。

这里就是不断从 DataLoader 取出 Batch。

假设 DataLoader 每次返回：

```python
(
    tensor([[0.8, 0.6],
            [0.9, 0.3]]),

    tensor([0, 1])
)
```

那么 Python 会自动解包：

```python
batch_x = 第一个Tensor

batch_y = 第二个Tensor
```

所以这里才可以写：

```python
for batch_x, batch_y in loader:
```

这里再强调一个之前容易混淆的细节：

**DataLoader 返回什么结构，取决于 Dataset 的 `__getitem__()` 返回什么结构，以及批处理整理方式。**

例如：

如果 Dataset 返回：

```python
return self.features[idx], self.labels[idx]
```

那么可以这样取：

```python
for batch_x, batch_y in loader:
```

如果 Dataset 返回：

```python
return {
    "text": text,
    "label": label
}
```

那么可以这样取：

```python
for batch in loader:

    batch_text = batch["text"]
    batch_label = batch["label"]
```

这两种写法都正确，只是数据结构不同。

---

## 3.3 训练指标：Loss 和 Accuracy

### Loss

Loss 衡量模型预测与真实答案之间的差异。

通常我们希望 Loss 总体下降。

### Accuracy

Accuracy 是分类准确率。

公式：

$$
Accuracy=\frac{\text{预测正确的样本数}}{\text{总样本数}}
$$

例如：

100 条样本中，模型预测正确了 85 条。

那么：

$$
Accuracy=\frac{85}{100}=85\%
$$

PyTorch：

```python
predictions = torch.argmax(logits, dim=1)

correct = (predictions == batch_y).sum().item()
```

拆开来看：

```python
predictions == batch_y
```

会逐元素比较：

```python
predictions = tensor([0, 1, 2, 0])
batch_y     = tensor([0, 1, 1, 0])
```

得到：

```python
tensor([True, True, False, True])
```

然后：

```python
.sum()
```

把 `True` 当作 1、`False` 当作 0，得到：

```python
tensor(3)
```

最后：

```python
.item()
```

转换为 Python 数字：

```python
3
```

---

# 第四步：从零构建完整训练代码

今天我们继续水果分类任务，但是为了方便训练和观察，我们构造一个稍微大一点的合成数据集。

这次不再手动写 4 条数据，而是生成 300 条训练样本。

## 4.1 构造数据集

```python
import torch

torch.manual_seed(42)

num_per_class = 100

class_0 = torch.randn(100, 2) * 0.25 + torch.tensor([1.0, 1.0])

class_1 = torch.randn(100, 2) * 0.25 + torch.tensor([3.0, 1.0])

class_2 = torch.randn(100, 2) * 0.25 + torch.tensor([2.0, 3.0])
```

这里有三个新知识点。

**第一个：`torch.randn(100, 2)`**

生成一个形状为 `[100, 2]` 的随机张量，元素服从标准正态分布。

**第二个：`* 0.25`**

将随机数据缩小，让每一类的数据更加集中。

**第三个：`+ torch.tensor([1.0, 1.0])`**

利用广播机制，将一组随机数据整体移动到指定的中心位置。

所以三种水果的特征会形成三个不同的分布：

**合成数据分布示意图**

| 类别 | 特征1 | 特征2 |
|---|---|---|
| 苹果 | 0.7 | 1.1 |
| 苹果 | 0.85 | 0.8 |
| 苹果 | 1 | 1 |
| 苹果 | 1.2 | 0.9 |
| 苹果 | 1.1 | 1.2 |
| 香蕉 | 2.7 | 1.1 |
| 香蕉 | 2.9 | 0.8 |
| 香蕉 | 3 | 1 |
| 香蕉 | 3.2 | 1.2 |
| 香蕉 | 3.1 | 0.9 |
| 橙子 | 1.7 | 3 |
| 橙子 | 1.9 | 2.8 |
| 橙子 | 2 | 3.1 |
| 橙子 | 2.2 | 3 |
| 橙子 | 2.1 | 3.2 |

原图的示意点按表格保留，三类中心约为 (1,1)、(3,1)、(2,3)。

注意：这些只是用于教学的人工合成特征，并不代表真实水果的甜度或重量。

我们再把数据合并：

```python
X = torch.cat(
    [class_0, class_1, class_2],
    dim=0
)
```

`torch.cat()` 的作用是拼接 Tensor。

这里：

```python
class_0.shape  # [100, 2]
class_1.shape  # [100, 2]
class_2.shape  # [100, 2]
```

沿 `dim=0` 拼接：

```python
X.shape  # [300, 2]
```

接着生成真实标签：

```python
y = torch.cat([
    torch.zeros(100, dtype=torch.long),
    torch.ones(100, dtype=torch.long),
    torch.full((100,), 2, dtype=torch.long)
])
```

这里：

- `torch.zeros(100)`：生成 100 个 0。
- `torch.ones(100)`：生成 100 个 1。
- `torch.full((100,), 2)`：生成 100 个 2。

因此：

```python
y.shape  # [300]
```

数据已经准备好了。

---

## 4.2 创建 Dataset 和 DataLoader

这里使用 PyTorch 自带的 `TensorDataset`。

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

你之前自己写过 Dataset。

今天为了把重点放在训练循环上，我们使用 `TensorDataset`。

它的作用就是：将多个 Tensor 按照第一维组织成样本。

例如：

```python
dataset[0]
```

返回：

```python
(
    X[0],
    y[0]
)
```

也就是说：

```python
features, label = dataset[0]
```

可以直接解包。

然后：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

这里：

- `batch_size=32`：每个 Batch 最多 32 条样本。
- `shuffle=True`：每次遍历时打乱样本顺序。

300 条数据、Batch Size 为 32：

$$
\left\lceil300/32\right\rceil=10
$$

因此，一个 Epoch 包含 10 个 Batch，最后一批只有 12 条。

---

## 4.3 定义模型

```python
import torch.nn as nn

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
```

网络结构：

```text
输入数据 [B, 2]
       ↓
Linear(2, 16)
       ↓
ReLU
       ↓
Linear(16, 3)
       ↓
Logits [B, 3]
```

这里 B 就是当前 Batch 的实际大小。

通常是 32，最后一个 Batch 是 12。

---

## 4.4 创建优化器和损失函数

```python
model = FruitClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

你在 Day 9 和 Day 10 已经学过。

- `model`：分类模型。
- `criterion`：交叉熵损失函数。
- `optimizer`：SGD 优化器。

接下来进入今天最重要的部分。

---

## 4.5 编写训练循环

先看基础版本：

```python
num_epochs = 10

for epoch in range(num_epochs):

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()
```

你现在应该能完全理解这几行。

但它还缺少训练指标统计。

我们继续完善。

---

## 4.6 统计每个 Epoch 的 Loss

这是训练代码中非常重要的工程细节。

先看完整写法：

```python
num_epochs = 10

for epoch in range(num_epochs):

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

这里有个你必须掌握的细节：

**为什么是 `loss.item() * batch_size`，而不是直接 `total_loss += loss.item()`？**

因为：

```python
nn.CrossEntropyLoss()
```

默认返回当前 Batch 所有样本的平均 Loss。

假设：

- Batch 1：32 条，平均 Loss = 0.5。
- Batch 2：12 条，平均 Loss = 1.0。

如果直接对两个 Batch 的 Loss 求平均：

$$
\frac{0.5+1.0}{2}=0.75
$$

这个计算不准确，因为两批样本数量不同。

正确方法应该按照样本数量加权：

$$
\frac{0.5\times32+1.0\times12}{44}
\approx0.6364
$$

因此：

```python
total_loss += loss.item() * batch_size
```

是为了得到所有样本的 Loss 总和。

最后：

```python
avg_loss = total_loss / total_samples
```

才是整个 Epoch 的平均 Loss。

这就是实际工程代码为什么需要这么写。

---

## 4.7 同时统计 Accuracy

我们在代码中增加：

```python
total_correct = 0
```

每个 Batch 预测完成后：

```python
predictions = torch.argmax(logits, dim=1)

correct = (predictions == batch_y).sum().item()

total_correct += correct
```

最后：

```python
accuracy = total_correct / total_samples
```

这样就能得到整个 Epoch 的训练准确率。

完整版本如下：

```python
num_epochs = 10

for epoch in range(num_epochs):

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )
```

现在，我们就有了一个能够自动训练多个 Epoch、统计 Loss 和 Accuracy 的完整训练循环。

你可能会注意到：

```python
f"Loss: {avg_loss:.4f}"
```

这里 `:.4f` 表示保留四位小数。

而：

```python
f"Accuracy: {accuracy:.2%}"
```

这里 `:.2%` 表示以百分比形式输出，并保留两位小数。

例如：

```python
accuracy = 0.8567
```

输出：

```text
Accuracy: 85.67%
```

另外，当前 Accuracy 统计的是每个 Batch 在参数更新前产生的预测，因此它反映的是这个 Epoch 训练过程中的平均表现，而不是 Epoch 结束后固定模型在整个训练集上的一次重新评估。

这是正常且常见的训练指标统计方式。

---

# 第五步：Day 11 完整可运行代码

文件名称：`train_loop.py`

下面这份代码包含今天所有的重要内容。你可以对照敲一遍，重点关注两个 `for` 循环、指标累加和 Shape。

```python

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

# ============================================
# 2. 创建合成训练数据
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

# 合并输入特征
X = torch.cat(
    [class_0, class_1, class_2],
    dim=0
)

# 生成对应标签
y = torch.cat([
    torch.zeros(num_per_class, dtype=torch.long),
    torch.ones(num_per_class, dtype=torch.long),
    torch.full((num_per_class,), 2, dtype=torch.long)
])

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================
# 3. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("样本总数:", len(dataset))
print("每个 Epoch 的 Batch 数:", len(loader))

# ============================================
# 4. 定义分类模型
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
# 5. 创建模型、损失函数和优化器
# ============================================

model = FruitClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# ============================================
# 6. 训练参数
# ============================================

num_epochs = 10

# ============================================
# 7. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    # 切换到训练模式
    model.train()

    # 每个 Epoch 开始时清零统计指标
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 遍历当前 Epoch 的所有 Batch
    for batch_index, (batch_x, batch_y) in enumerate(loader):

        # 1. 清零旧梯度
        optimizer.zero_grad()

        # 2. Forward
        logits = model(batch_x)

        # 3. Calculate Loss
        loss = criterion(logits, batch_y)

        # 4. Backward
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

        # 6. 统计当前 Batch 的样本数量
        batch_size = batch_x.size(0)

        # 7. 累计 Loss
        total_loss += loss.item() * batch_size

        # 8. 得到预测类别
        predictions = torch.argmax(logits, dim=1)

        # 9. 累计预测正确数量
        correct = (predictions == batch_y).sum().item()

        total_correct += correct

        # 10. 累计样本数量
        total_samples += batch_size

        # 11. 第一轮打印部分 Batch 信息
        if epoch == 0 and batch_index < 2:

            print(f"\nBatch {batch_index + 1}")

            print("batch_x shape:", batch_x.shape)
            print("batch_y shape:", batch_y.shape)

            print("logits shape:", logits.shape)
            print("当前 Batch Loss:", loss.item())

    # ========================================
    # 8. 计算 Epoch 指标
    # ========================================

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )

# ============================================
# 9. 训练结束
# ============================================

print("\n训练完成！")
```

这份代码只需要安装 PyTorch，就可以在 CPU 上运行，不需要 GPU。

运行后，你会看到类似：

```text
X shape: torch.Size([300, 2])
y shape: torch.Size([300])

样本总数: 300
每个 Epoch 的 Batch 数: 10

Batch 1
batch_x shape: torch.Size([32, 2])
batch_y shape: torch.Size([32])
logits shape: torch.Size([32, 3])
当前 Batch Loss: ...

Epoch [1/10] Loss: ... Accuracy: ...
Epoch [2/10] Loss: ... Accuracy: ...
...
Epoch [10/10] Loss: ... Accuracy: ...

训练完成！
```

实际 Loss 和 Accuracy 由运行结果决定。

对于这种简单、类别容易区分的合成数据，正常情况下训练 Loss 应该总体下降，准确率也应该逐渐提高。

---

# 第六步：必须掌握的工程细节

在进入 Day 12 之前，还有三个很重要的细节。

### 1\. `model.train()` 到底做什么？

你在完整代码里看到了：

```python
model.train()
```

它表示把模型切换到训练模式。

**注意：`model.train()` 本身不会执行训练，也不会更新参数。**

它主要影响 Dropout、BatchNorm 等部分模块的行为。

真正的参数更新还是：

```python
loss.backward()
optimizer.step()
```

我们当前的网络只有 Linear 和 ReLU，因此调用 `model.train()` 不会改变这些层的计算方式。

但在实际工程中，仍然应该养成显式设置训练模式的习惯。

### 2\. 为什么每个 Epoch 都要重新初始化指标？

```python
total_loss = 0.0
total_correct = 0
total_samples = 0
```

因为我们想统计的是当前 Epoch 的训练表现。

如果把这些变量放在整个训练循环外部，那么它们就会不断累积所有 Epoch 的数据。

这样计算出来的结果就不是当前这一轮的指标了。

所以指标初始化必须放在外层循环内部、内层循环之前。

### 3\. 为什么 `zero_grad()` 放在 Batch 循环内部？

因为我们当前希望每处理一个 Batch，就执行一次参数更新。

每一轮都应该使用当前 Batch 新计算出的梯度。

因此：

```python
for batch_x, batch_y in loader:

    optimizer.zero_grad()

    logits = model(batch_x)

    loss = criterion(logits, batch_y)

    loss.backward()

    optimizer.step()
```

这是普通训练循环的标准写法。

不过，有一种特殊的情况：Gradient Accumulation（梯度累积）。

如果希望每 4 个 Batch 才更新一次参数，就不能每个 Batch 都清零梯度。

这时需要调整 `zero_grad()` 和 `step()` 的执行时机，并正确处理 Loss 缩放。

你之前学过梯度累积的基本原理。之后我们会在工业训练技巧模块中进一步实现完整代码。

---

# 第七步：今日知识巩固

下面用几道题检验你是否真正掌握训练循环。

原窗口此处为交互检测题，题目与作答未提供可读取内容。

---

# Day 11 最终总结

今天最重要的是理解：**完整模型训练不是简单重复 Forward，而是通过 Epoch 和 Batch 组织数据，并持续执行梯度更新。**

你需要记住这个最精简的代码模板：

```python
for epoch in range(num_epochs):

    model.train()

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()
```

以及三个核心概念：

| 概念 | 你应该记住的内容 |
|--------|--------|
| Batch | 一次处理的一组样本 |
| Iteration | 一次训练循环，通常对应一个 Batch |
| Epoch | 遍历完整训练数据一轮 |

再加上两个工程能力：

- 正确统计每个 Epoch 的平均 Loss。
- 正确计算分类 Accuracy。

现在，你已经能够把 PyTorch 的基本训练过程真正组织起来了。

**Day 11 到这里结束。**

下一天是模块三最后一天：

**Day 12：验证与泛化（Validation & Generalization）**

我们将解决一个非常关键的实际问题：

假设你的模型在训练集上准确率已经达到 99%，那么它面对从未见过的数据时，是否也能达到 99%？

这就要学习 Train / Validation / Test、过拟合、欠拟合、`model.eval()`、`torch.no_grad()` 和 Early Stopping，并最终写出完整的 `train_val.py`。

---

## 归档代码的整理说明

code.py 来自本地 train_loop.py。将 torch.manual_seed=42 修正为 torch.manual_seed(42)，恢复设置种子的函数调用，并将样本 Shape 注释更正为 [300,2]。模型结构、数据生成、训练轮数和优化器设置保留原练习。
