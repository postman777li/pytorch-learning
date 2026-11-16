# Day 12：验证与泛化（Validation & Generalization）

> 来源为《学习Day9前向与损失》，保留原讲解顺序、设计动机、例子、反例、推导与逐步代码说明，交互图表转为静态表格，公式适配 GitHub。归档程序见 [code.py](code.py)。

(Day 12 / 12) (模块三 · 最后一天)

今天我们要完成一个非常重要的跨越：**从“模型能够训练”到“知道模型训练得究竟好不好”。**

在 Day 11，你已经写出了完整训练循环：

```python
for epoch in range(num_epochs):

    model.train()

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()
```

但是这里有一个问题：

假设你训练了 100 个 Epoch，模型在训练集上的 Accuracy 达到 99.9%。

**这能证明你的模型已经足够优秀了吗？**

不能。

因为模型可能只是在记忆训练数据，并没有真正掌握能够应用到新数据上的规律。

今天我们按照之前的五步学习法：

1. 具体例子引入
2. 理解设计动机
3. 专业概念和原理
4. 代码逐步实践
5. 完整可运行的 `train_val.py`

最后还会讲清楚模型保存、Early Stopping 和测试集使用规则。这些都是实际算法工程中不可缺少的能力。

---

# 第一步：从一个生活例子理解验证

假设你准备参加数学考试。

老师给了你 100 道练习题。

你把这 100 道题反复做了 20 遍。

最后：

- 练习题正确率：100%
- 考试新题正确率：60%

为什么？

因为你可能记住了练习题的答案，却没有真正掌握解题规律。

这对应深度学习中的一个重要现象：

**过拟合（Overfitting）。**

如果只用做过的练习题衡量学习水平，就容易产生误判。

所以，我们需要把题目分成三部分。

**假设一共有 1,000 道题**

70%

20%

10%

**训练集（Train）· 700 道**

平时做的练习题，用来真正学习和调整参数

**验证集（Validation）· 200 道**

模拟考试，用来检查学习效果、选择训练方案

**测试集（Test）· 100 道**

最终考试，用来评估确定好的模型在新题上的表现

70% / 20% / 10% 只是示例比例，不是所有任务必须遵守的固定规则。

这个例子就是今天最核心的思想。

**训练集用于学习，验证集用于选择，测试集用于最终评估。**

---

# 第二步：为什么需要验证机制？

## 2.1 为什么训练集准确率不能说明一切？

我们继续使用 Day 11 的水果分类模型。

假设训练 20 个 Epoch，得到：

| Epoch | Train Accuracy | Validation Accuracy |
|--------|--------|--------|
| 1 | 65% | 62% |
| 5 | 85% | 83% |
| 10 | 95% | 90% |
| 15 | 99% | 84% |
| 20 | 100% | 76% |

你发现了什么？

训练准确率一直在提高。

但是验证准确率却在第 10 轮之后开始下降。

也就是说：

**模型在训练集上越来越优秀，但面对没有参与参数更新的新数据时，表现反而变差了。**

这就是典型的过拟合迹象。

## 2.2 为什么不能只在训练结束后验证一次？

因为我们还希望知道：

- 模型训练多少个 Epoch 最合适？
- 哪一个 Epoch 的模型表现最好？
- 是否应该提前停止训练？
- 学习率、模型结构等是否需要调整？

因此，实际训练中经常会在每个 Epoch 结束后执行一次 Validation。

**Epoch 1：训练**

使用训练集更新模型参数

**Validation 1：验证**

只评估，不更新模型参数

**Epoch 2：继续训练**

模型参数继续更新

**Validation 2：再次验证**

观察模型的泛化表现

持续重复，选择验证表现最好的模型

注意：验证集虽然不直接参与梯度更新，但我们会根据验证结果选择模型和调整超参数。所以它也会间接影响开发过程。

测试集才应该留到最终阶段使用。

---

# 第三步：专业概念与原理

今天需要掌握六个概念。

## 3.1 Train / Validation / Test

| 数据集 | 是否更新梯度 | 主要用途 |
|--------|--------|--------|
| Train | 是 | 学习模型参数 |
| Validation | 否 | 模型选择、调参、Early Stopping |
| Test | 否 | 最终独立评估 |

需要特别注意：验证集和测试集不应该因为每个 Epoch 的训练循环而参与参数更新。

而且数据划分不能随意造成数据泄漏。

例如，同一个人的几乎重复的照片，如果分别进入训练集和测试集，就可能让测试结果虚高。

在真实业务中，还需要根据具体问题使用时间划分、按用户分组划分等方式。

## 3.2 Generalization：泛化能力

专业定义：

**泛化能力指模型在未参与训练的新数据上保持良好预测表现的能力。**

假设：

- 模型 A：训练准确率 99%，验证准确率 70%。
- 模型 B：训练准确率 94%，验证准确率 92%。

通常而言，在验证数据具有代表性的前提下，模型 B 更值得选择。

因为我们的最终目标不是把训练集记得多么牢固，而是让模型能够处理未来遇到的新数据。

## 3.3 Overfitting：过拟合

过拟合的常见表现：

- Train Loss 持续下降。
- Validation Loss 先下降，后来上升。
- 训练集和验证集之间出现明显性能差距。

**观察过拟合：Train Loss 与 Val Loss**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.1 | 1.14 |
| 2 | 0.85 | 0.91 |
| 3 | 0.63 | 0.7 |
| 4 | 0.47 | 0.55 |
| 5 | 0.36 | 0.47 |
| 6 | 0.27 | 0.43 |
| 7 | 0.2 | 0.46 |
| 8 | 0.15 | 0.51 |
| 9 | 0.11 | 0.59 |
| 10 | 0.08 | 0.7 |

以上保留原图示意数值，用于观察过拟合。

第 6 轮附近，Validation Loss 达到最低值。之后训练 Loss 继续下降，验证 Loss 却开始升高。

常见应对方法包括正则化、Dropout、数据增强、减少模型复杂度、增加高质量训练数据，以及今天要学习的 Early Stopping。

但注意：一次 Validation Loss 上升不一定说明已经过拟合，也可能是正常波动。

## 3.4 Underfitting：欠拟合

欠拟合和过拟合正好不同。

欠拟合是模型连训练数据中的主要规律都没有学好。

例如：

- Train Accuracy = 55%
- Validation Accuracy = 53%

如果这是一个简单的三分类任务，这两个准确率可能表明模型学习不足。

常见原因：

- 模型表达能力不足。
- 训练轮次不够。
- 学习率设置不合适。
- 特征不足。
- 正则化过强。

### 判断模型状态

| 状态 | Train 表现 | Validation 表现 |
|--------|--------|--------|
| 欠拟合 | 差 | 差 |
| 拟合较好 | 好 | 好，差距较小 |
| 过拟合 | 很好 | 明显变差 |

注意这只是典型模式，不能仅凭两个准确率数字就下最终结论。

## 3.5 model.eval() 与 torch.no\_grad()

这是今天最重要的 PyTorch 工程知识点之一。

训练时：

```python
model.train()
```

验证时：

```python
model.eval()

with torch.no_grad():
    logits = model(x)
```

你要特别区分两者。

**`model.eval()`：切换模型的运行模式。**

它会影响部分神经网络模块，例如：

- Dropout：训练时随机屏蔽部分特征，验证时不随机屏蔽。
- BatchNorm：训练和评估时通常采用不同的统计行为。

但它不会自动关闭梯度计算。

**`torch.no_grad()`：关闭当前上下文中的自动求导图记录。**

因为验证不需要 `backward()`，所以不需要保存反向传播所需的计算图。

| 操作 | 改变训练/评估模式 | 关闭梯度记录 |
|--------|--------|--------|
| `model.train()` | 是，训练模式 | 否 |
| `model.eval()` | 是，评估模式 | 否 |
| `torch.no_grad()` | 否 | 是 |

**最推荐的习惯是：训练时调用 `model.train()`，验证和测试时同时使用 `model.eval()` 与 `torch.no_grad()`。**

## 3.6 Early Stopping：提前停止训练

假设我们计划训练 100 个 Epoch。

但是模型在第 20 轮就达到了最好的验证表现，之后没有进一步提升。

那为什么还要继续训练到 100 轮？

我们可以使用 Early Stopping。

它的基本思想是：

**如果验证指标在连续若干轮中没有达到规定的改善幅度，就停止训练。**

例如：

```python
patience = 3
```

表示允许连续 3 轮没有改善。

**Early Stopping 静态示例**

取 patience=3，以下序列用于复现原文的停止过程。

| Epoch | Val Loss（示意） | 连续未改善轮数 | 结果 |
|---|---|---|---|
| 1 | 0.80 | 0 | 保存 |
| 2 | 0.60 | 0 | 保存 |
| 3 | 0.48 | 0 | 保存最佳状态 |
| 4 | 0.49 | 1 | 等待 |
| 5 | 0.50 | 2 | 等待 |
| 6 | 0.51 | 3 | 停止并恢复第3轮 |
| 7 | 0.47 | — | 假设继续可能达到，但已停止 |

例如，当 `patience=3`，第 3 轮达到最佳值 0.48 后，第 4、5、6 轮都没有改善，于是第 6 轮结束时停止训练。

即使第 7 轮原本可能出现更低的损失，也不会继续执行。

因此，Early Stopping 是一种实用的启发式策略，并不保证找到全局最优模型。

还有一个重要细节：

**停止训练时的最后一个模型，不一定是验证表现最好的模型。**

所以实际工程里通常还需要保存最佳模型参数。

---

# 第四步：从代码开始实现验证流程

下面继续使用 Day 11 的水果分类任务。

这次为了理解验证，先把全部数据划分成 Train、Validation 和 Test。

## 4.1 划分数据集

假设我们已经有：

```python
dataset = TensorDataset(X, y)
```

一共有 300 条样本。

我们希望：

- 训练集：210 条
- 验证集：60 条
- 测试集：30 条

使用：

```python
from torch.utils.data import random_split

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [210, 60, 30],
    generator=torch.Generator().manual_seed(42)
)
```

这里有一个新的函数：

`random_split()`

它用于将 Dataset 随机拆分成多个互不重叠的子集。

其中：

```python
[210, 60, 30]
```

表示三个子集的样本数量。

而：

```python
generator=torch.Generator().manual_seed(42)
```

用于控制本次随机划分，使结果可复现。

注意：`random_split()` 本身不保证分层抽样，因此对于类别不平衡的数据，还需要采用更合适的数据划分方法。

## 4.2 创建三个 DataLoader

```python
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

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

你应该能解释这三行。

训练集：

```python
shuffle=True
```

一般希望训练期间打乱顺序，降低固定数据顺序带来的影响。

验证集和测试集：

```python
shuffle=False
```

通常不需要打乱，因为它们不进行参数更新。是否打乱本身不决定验证结果是否有效，只是评估时保持固定顺序更方便排查问题。

## 4.3 先写一个验证函数

这是今天最值得你掌握的代码。

```python
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch_x, batch_y in loader:

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)

            total_correct += (
                predictions == batch_y
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy
```

这个函数和 Day 11 的训练代码很像。

但最重要的区别是：

**没有 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。**

因为我们只是评估模型，不更新权重。

### 为什么要写成函数？

因为 Validation 和 Test 都需要使用相同的评估逻辑。

如果分别写两套代码，不仅重复，而且更容易出现不一致。

所以我们封装：

```python
def evaluate(model, loader, criterion):
```

其中三个参数分别代表：

- `model`：需要评估的神经网络。
- `loader`：需要评估的数据集对应的 DataLoader。
- `criterion`：计算损失的函数。

以后只需要：

```python
val_loss, val_acc = evaluate(
    model,
    val_loader,
    criterion
)
```

或者：

```python
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion
)
```

你可能会问：

**为什么 `evaluate()` 可以同时返回两个值？**

因为：

```python
return avg_loss, accuracy
```

在 Python 中会返回一个包含这两个值的元组。

然后：

```python
val_loss, val_acc = evaluate(...)
```

通过元组解包分别接收两个结果。

这和之前：

```python
for batch_x, batch_y in loader:
```

涉及的是同一种 Python 解包机制。

---

# 第五步：Day 12 完整可运行代码

现在我们把所有内容组合起来。

文件名称：

`train_val.py`

这次的代码会真正完成：

- 创建数据集
- 划分 Train / Validation / Test
- 定义模型
- 训练多个 Epoch
- 每轮执行 Validation
- 统计 Loss 和 Accuracy
- 使用 Early Stopping
- 保存最佳模型参数
- 恢复最佳模型
- 最终执行 Test

为了让 Early Stopping 的行为更容易观察，代码中使用一个合理的 `patience`，但不保证每次一定提前停止。这取决于真实训练曲线。

```python

import copy

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

# ============================================
# 2. 创建合成数据
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

X = torch.cat(
    [class_0, class_1, class_2],
    dim=0
)

y = torch.cat([
    torch.zeros(num_per_class, dtype=torch.long),
    torch.ones(num_per_class, dtype=torch.long),
    torch.full((num_per_class,), 2, dtype=torch.long)
])

dataset = TensorDataset(X, y)

# ============================================
# 3. 划分训练、验证和测试数据
# ============================================

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [210, 60, 30],
    generator=torch.Generator().manual_seed(42)
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

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ============================================
# 4. 定义模型
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
# 6. 定义评估函数
# ============================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch_x, batch_y in loader:

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions == batch_y
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy

# ============================================
# 7. Early Stopping 配置
# ============================================

num_epochs = 50

patience = 5

min_delta = 0.0001

best_val_loss = float("inf")

best_model_state = None

best_epoch = 0

epochs_without_improvement = 0

# ============================================
# 8. 完整训练与验证循环
# ============================================

for epoch in range(num_epochs):

    # 切换训练模式
    model.train()

    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    # ---------- Train ----------

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        total_train_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_train_correct += (
            predictions == batch_y
        ).sum().item()

        total_train_samples += batch_size

    train_loss = (
        total_train_loss / total_train_samples
    )

    train_acc = (
        total_train_correct / total_train_samples
    )

    # ---------- Validation ----------

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion
    )

    # ---------- 打印指标 ----------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.2%}"
    )

    # ---------- Early Stopping ----------

    if val_loss < best_val_loss - min_delta:

        best_val_loss = val_loss

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print("\nEarly Stopping triggered!")
            break

# ============================================
# 9. 恢复最佳模型参数
# ============================================

model.load_state_dict(best_model_state)

print("\nBest Epoch:", best_epoch)

print("Best Val Loss:", best_val_loss)

# ============================================
# 10. 在测试集上进行最终评估
# ============================================

test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion
)

print("\n========== Final Test ==========")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# ============================================
# 11. 保存最佳模型
# ============================================

torch.save(
    model.state_dict(),
    "best_fruit_model.pth"
)

print("\n最佳模型已保存！")
```

运行后你会看到类似结构：

```text
Train samples: 210
Val samples: 60
Test samples: 30

Epoch [1/50] Train Loss: ... Train Acc: ... | Val Loss: ... Val Acc: ...
Epoch [2/50] Train Loss: ... Train Acc: ... | Val Loss: ... Val Acc: ...
Epoch [3/50] Train Loss: ... Train Acc: ... | Val Loss: ... Val Acc: ...
...

Best Epoch: ...
Best Val Loss: ...

========== Final Test ==========
Test Loss: ...
Test Accuracy: ...

最佳模型已保存！
```

这里不预设具体数值，因为训练指标需要由实际运行结果决定。

---

# 第六步：完整代码中必须掌握的工程知识

这部分是今天真正提升工程能力的关键。

## 6.1 `float("inf")` 是什么？

```python
best_val_loss = float("inf")
```

这是 Python 中的正无穷大。

为什么要初始化为正无穷大？

因为我们希望第一次计算得到的有限 Validation Loss 一定小于它。

例如：

```python
best_val_loss = float("inf")

val_loss = 0.8

if val_loss < best_val_loss:
    print("找到更好的模型")
```

条件成立。

之后：

```python
best_val_loss = val_loss
```

就把当前最佳损失更新成 0.8。

以后只有出现更小的 Validation Loss，才会继续更新。

## 6.2 为什么用 `copy.deepcopy()` 保存最佳参数？

```python
best_model_state = copy.deepcopy(
    model.state_dict()
)
```

先理解：

```python
model.state_dict()
```

它返回模型的状态字典，其中包含权重、偏置，以及其他需要保存的持久状态。

例如：

```python
{
    "network.0.weight": tensor(...),
    "network.0.bias": tensor(...),
    "network.2.weight": tensor(...),
    "network.2.bias": tensor(...)
}
```

但是，如果直接：

```python
best_model_state = model.state_dict()
```

其中的 Tensor 仍然可能与模型底层存储共享数据。

随着后续训练继续更新，保存的所谓“最佳参数”也可能跟着改变。

因此我们使用：

```python
copy.deepcopy(...)
```

创建独立副本。

这样才能真正保留某个 Epoch 的参数快照。

## 6.3 `model.load_state_dict()` 是什么？

```python
model.load_state_dict(best_model_state)
```

它用于将保存的参数和状态加载回当前模型。

举个例子：

- Epoch 10：Validation Loss = 0.30
- Epoch 11：Validation Loss = 0.35
- Epoch 12：Validation Loss = 0.39

如果第 12 轮触发停止，我们不希望使用第 12 轮的权重。

我们希望回到第 10 轮。

因此：

```python
model.load_state_dict(best_model_state)
```

就是恢复此前保存的最佳状态。

注意，最佳的判断依据是我们选择的 Validation Loss，并不保证它在 Test 集上也是最好的。

## 6.4 `torch.save()` 是什么？

```python
torch.save(
    model.state_dict(),
    "best_fruit_model.pth"
)
```

这行代码把当前模型参数保存到文件。

其中：

- `model.state_dict()`：需要保存的模型状态。
- `"best_fruit_model.pth"`：保存路径和文件名。

`.pth` 是 PyTorch 中常见的模型权重文件扩展名。

以后如果想重新使用模型：

```python
model = FruitClassifier()

state = torch.load(
    "best_fruit_model.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()
```

就可以恢复之前训练好的模型参数。

这里的 `weights_only=True` 用于采用受限的权重加载方式，但仍应只加载可信来源的模型文件。

另外，今天保存的是**模型状态**，不是完整的训练检查点。

如果以后希望中断训练再继续，通常还需要保存优化器状态、Epoch、学习率调度器状态等。

这些属于后续工业级训练工程内容。

## 6.5 `break` 是什么？

```python
if epochs_without_improvement >= patience:
    break
```

`break` 是 Python 的循环控制语句。

它会立即退出所在的最近一层循环。

在今天的代码中，`break` 位于外层 Epoch 循环中，因此会结束整个训练循环。

注意它不会直接结束整个 Python 程序。

所以后面的：

```python
model.load_state_dict(best_model_state)
```

以及 Test、模型保存仍然会执行。

## 6.6 `min_delta` 是什么？

```python
min_delta = 0.0001
```

我们不希望 Validation Loss 只降低了一个极小数值，就永远被认为出现了明显改善。

所以设置最小改善阈值：

```python
if val_loss < best_val_loss - min_delta:
```

只有改善幅度超过 `min_delta`，才会重置等待计数。

本例为了让 Early Stopping 逻辑保持简单，只有满足这个条件才更新保存的最佳模型。

实际工程中也可以把“保存历史最低 Loss”和“判断是否达到有意义的改善”设计成两套独立逻辑。

---

# 第七步：今天需要建立的完整训练架构

到这里，你已经学完模块三的整个训练过程。

**Dataset / DataLoader**

读取并组织训练数据

Train：训练阶段

1. 清零旧梯度
2. Forward
3. Calculate Loss
4. Backward
5. Optimizer Step

每个 Batch 重复，完成一个 Epoch

**Validation：验证阶段**

model.eval() + torch.no\_grad()

计算 Val Loss / Val Accuracy

选择最佳模型，判断是否 Early Stopping

继续下一个 Epoch，直到训练结束或提前停止

**恢复最佳模型参数**

load\_state\_dict(best\_model\_state)

**Test：最终测试**

使用独立测试集进行最终评估

**保存训练完成的模型**

torch.save(model.state\_dict(), ...)

这是你以后学习模型训练、微调 LLM 时可以不断复用的整体框架。

---

# 第八步：Day 12 知识验收

下面的五个问题专门检查你是否理解了验证与泛化的核心区别。

原窗口此处为交互检测题，题目与作答未提供可读取内容。

---

# 模块三：Day 9–12 总结

**模型训练完整流程** (4 / 4 天课程已讲解)

| **Day 9** | Forward & Loss<br><br>模型产生预测，计算损失 |
|--------|--------|
| **Day 10** | Backward & Optimizer<br><br>计算梯度，更新参数 |
| **Day 11** | Train Loop<br><br>通过 Epoch、Batch 组织完整训练 |
| **Day 12** | Validation & Generalization<br><br>验证性能，选择最佳模型，防止过拟合 |

学完今天，你应该能够理解并独立实现：

```python
for epoch in range(num_epochs):

    # Train
    model.train()

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

    # Validation
    val_loss, val_acc = evaluate(
        model, val_loader, criterion
    )
```

**这段代码是你后面学习大模型 SFT、LoRA 微调和其他训练工程的基础。**

从下一模块开始，我们就可以进一步深入 SGD、Adam、AdamW、学习率调度、梯度裁剪、梯度累积以及混合精度训练，把当前这套基础训练代码逐步升级成更接近工业实践的训练代码。

不过今天先以你能够亲自敲出并理解 `train_val.py` 为目标。这样 Day 12 才算真正掌握，而不只是看过。

---

## 归档代码的整理说明

code.py 来自本地 train_val.py。将 torch.manual_seed=42 修正为 torch.manual_seed(42)，恢复设置种子的函数调用，并将样本 Shape 注释更正为 [300,2]。权重路径改为脚本旁的 best_fruit_model.pth。模型结构、数据生成、训练轮数和优化器设置保留原练习。
