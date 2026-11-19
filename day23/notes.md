好，我们进入 **模块六 · Day 23：模型评估（Model Evaluation）**。

今天继续使用你之前的 `FruitClassifier` 水果分类项目，不换数据集、不换模型。我们在 Day 22 的基础上，把原来的训练代码升级成一个具备规范评估能力的训练项目。

今天非常重要，因为在真实的大模型训练和算法工程中：

**训练 Loss 下降，不代表模型真的学得好；训练集表现好，也不代表模型在新数据上表现好。**

今天我们要解决的就是这个问题。

# Day 23：模型评估

(今日目标) 独立实现训练集、验证集、测试集划分，掌握分类指标计算，并能够根据验证集表现选择最佳模型。

今天学习七个知识点：

| 序号 | 知识点 | 今天要掌握什么 |
|--------|--------|--------|
| 1 | Train / Validation / Test | 三种数据集的区别 |
| 2 | Evaluation | 如何规范地评估模型 |
| 3 | Accuracy | 准确率 |
| 4 | Precision / Recall / F1 | 精确率、召回率、F1 |
| 5 | Perplexity | 困惑度及其在 LLM 中的作用 |
| 6 | Model Selection | 如何选择最佳模型 |
| 7 | Test Evaluation | 如何进行最终测试 |

---

# 第一步：具体例子引入

假设你训练了一个水果分类模型。

你给它看了 300 个水果：

- 100 个苹果
- 100 个香蕉
- 100 个橙子

模型训练完成后，你发现：

```text
训练集 Accuracy = 99%
```

是不是说明模型已经非常优秀了？

不一定。

我们举一个极端例子：

假设模型只是记住了训练数据中的所有水果，而没有真正学会水果的特征。

那么：

- 见过的水果：识别得非常准确。
- 没见过的水果：可能完全认不出来。

这就是机器学习中的一个重要问题：**过拟合（Overfitting）**。

> 原窗口此处为配图搜索组件，没有返回实际图片。下面用静态表格补充其表达的泛化概念。

| 现象 | 训练集 | 验证集 | 理解 |
|---|---|---|---|
| 欠拟合 | 表现较差 | 表现较差 | 尚未学好训练规律 |
| 拟合适当 | 表现好 | 表现好 | 对新数据也有效 |
| 过拟合 | 表现很好 | 表现较差 | 泛化存在问题 |

我们真正想要的是：

模型不仅能识别训练过程中见过的水果，还能识别从来没有见过的新水果。

因此，我们需要将数据划分为三部分。

## 1\. Train / Validation / Test

假设我们有 300 个水果样本。

可以这样划分：

**数据集划分**

# 70%

**Training**

210 个样本

# 15%

**Validation**

45 个样本

# 15%

**Test**

45 个样本

三者分别负责什么？

**① Training Set：训练集**

用于真正训练模型。

```python
logits = model(batch_x)
loss = criterion(logits, batch_y)
loss.backward()
optimizer.step()
```

模型通过训练集不断调整参数。

**② Validation Set：验证集**

用于观察模型在未参与训练的数据上的表现。

例如：

```text
Epoch 1: Val Loss = 0.85
Epoch 2: Val Loss = 0.63
Epoch 3: Val Loss = 0.42
Epoch 4: Val Loss = 0.38
Epoch 5: Val Loss = 0.45
Epoch 6: Val Loss = 0.57
```

你会发现：

Epoch 4 之后，Validation Loss 开始上升。

这可能意味着模型开始过拟合。

我们可以根据验证集的表现，决定使用哪一轮的模型。

**③ Test Set：测试集**

测试集是最终考试。

它不参与模型训练，也不参与超参数选择。

当我们完成模型训练、确定模型版本之后，才使用测试集进行最终评估。

为什么？

因为如果你反复根据测试集的结果修改模型，那么测试集实际上也参与了模型选择，就不再是独立的最终测试了。

记住这句话：

- Train：让模型学习。
- Validation：帮助我们选择模型。
- Test：检验模型最终的泛化能力。

---

# 第二步：理解设计动机

## 2\. 为什么不能只看 Accuracy？

假设有一个疾病筛查模型。

1000 个人中：

- 990 人没有疾病。
- 10 人有疾病。

模型非常懒，它直接把所有人都预测为没有疾病。

结果：

- 990 人预测正确。
- 10 人预测错误。

那么：

$$
Accuracy=\frac{990}{1000}=99\%
$$

准确率高达 99%。

但这个模型有用吗？

完全没有。

因为它一个真正有疾病的人都没有识别出来。

所以，我们需要其他指标。

今天重点学习：

- Precision：精确率
- Recall：召回率
- F1-score：F1 分数

这三个指标比 Accuracy 更能帮助我们理解模型究竟犯了什么错误。

# 第三步：专业概念与原理

## 3\. Accuracy：准确率

Accuracy 是最直观的分类指标。

$$
Accuracy=\frac{预测正确的样本数}{总样本数}
$$

例如：

总共有 100 个水果。

模型预测正确 90 个。

$$
Accuracy=\frac{90}{100}=90\%
$$

代码：

```python
predictions = torch.argmax(logits, dim=1)

correct = (predictions == labels).sum().item()

accuracy = correct / len(labels)
```

这里你之前已经学习过：

```python
torch.argmax(logits, dim=1)
```

得到每个样本预测概率最高的类别对应的索引（这里是 Logits 最大的位置）。

---

## 4\. Precision、Recall、F1：三个重要指标

这部分是今天的核心。

首先，我们需要理解四个概念：

| 名称 | 含义 |
|--------|--------|
| TP | 实际为正，预测也为正 |
| FP | 实际为负，却预测为正 |
| FN | 实际为正，却预测为负 |
| TN | 实际为负，预测也为负 |

例如：

我们要识别苹果，把苹果看作正类，其他水果看作负类。

假设有 100 个水果。

| 实际情况 | 模型预测 | 数量 |
|--------|--------|--------|
| 苹果 | 苹果 | 30 |
| 非苹果 | 苹果 | 10 |
| 苹果 | 非苹果 | 5 |
| 非苹果 | 非苹果 | 55 |

那么：

- TP = 30
- FP = 10
- FN = 5
- TN = 55

### 4.1 Precision：精确率

它回答的问题是：

**模型预测为苹果的水果中，究竟有多少真的属于苹果？**

$$
Precision=\frac{TP}{TP+FP}
$$

代入：

$$
Precision=\frac{30}{30+10}=75\%
$$

也就是说：

模型说有 40 个苹果，其中只有 30 个是真的。

Precision 越高，说明模型误报越少。

### 4.2 Recall：召回率

它回答的问题是：

**所有真正的苹果中，模型成功识别出了多少？**

$$
Recall=\frac{TP}{TP+FN}
$$

代入：

$$
Recall=\frac{30}{30+5}=85.71\%
$$

也就是说：

真正有 35 个苹果，模型找到了 30 个。

Recall 越高，说明模型漏报越少。

### 4.3 F1-score：综合指标

Precision 和 Recall 有时会出现矛盾。

例如：

- 模型预测得非常保守，Precision 很高，但漏掉很多目标，Recall 很低。
- 模型预测得非常激进，Recall 很高，但误报很多，Precision 很低。

F1 就是综合考虑两者的指标。

$$
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

代入：

$$
F1=2\times\frac{0.75\times0.8571}{0.75+0.8571}
$$

$$
F1\approx0.80
$$

**三个指标的区别**

**Precision** (少误报)

预测出来的正类，有多少是真的？

---

**Recall** (少漏报)

真正的正类，有多少被找到了？

---

**F1** (综合平衡)

Precision 和 Recall 综合表现如何？

## 5\. 多分类任务中的 Precision / Recall / F1

我们的水果分类有三个类别：

```text
0：苹果
1：香蕉
2：橙子
```

刚才讲的 Precision、Recall、F1 是针对二分类的。

多分类怎么办？

常见做法是：**把每个类别分别当作正类，其他类别都当作负类，计算指标。**

例如：

- 计算苹果的 Precision、Recall、F1。
- 计算香蕉的 Precision、Recall、F1。
- 计算橙子的 Precision、Recall、F1。

然后对各类别进行平均。

最常见的有两种。

**Macro Average：宏平均**

三个类别直接取算术平均：

$$
F1_{macro}=\frac{F1_0+F1_1+F1_2}{3}
$$

每个类别权重相同。

**Weighted Average：加权平均**

根据每个类别的真实样本数量进行加权。

$$
F1_{weighted}=
\frac{\sum_i n_iF1_i}{\sum_i n_i}
$$

其中 $n_i$ 是第 $i$ 类的真实样本数量。

区别：

- Macro：每个类别同等重要。
- Weighted：样本数量多的类别权重更大。

我们今天的水果数据每类数量相同，因此两者通常比较接近。

---

## 6\. Confusion Matrix：混淆矩阵

真正计算多分类 Precision、Recall、F1 时，我们通常会先得到混淆矩阵。

例如：

**混淆矩阵示例**

行代表真实类别，列代表预测类别

| 真实 \\ 预测 | 苹果 | 香蕉 | 橙子 |
|--------|--------|--------|--------|
| **苹果** | 30 | 2 | 1 |
| **香蕉** | 3 | 28 | 2 |
| **橙子** | 1 | 2 | 29 |

对角线上的数字代表预测正确的数量；其他位置代表预测错误。

例如苹果：

- TP = 30
- FP = 3 + 1 = 4
- FN = 2 + 1 = 3

所以：

$$
Precision_{apple}=\frac{30}{30+4}
$$

$$
Recall_{apple}=\frac{30}{30+3}
$$

这就是混淆矩阵与三个分类指标之间的关系。

## 7\. Perplexity：困惑度

这个指标需要特别注意，因为你未来的方向是 LLM。

Perplexity，简称 PPL，中文叫困惑度。

它主要用于语言模型，衡量模型预测下一个 Token 时的不确定程度。

公式：

$$
PPL=\exp(L)
$$

其中 $L$ 是平均 Token 级别的交叉熵损失。

例如：

| 平均 Token Loss | Perplexity |
|--------|--------|
| 0.5 | 1.65 |
| 1.0 | 2.72 |
| 2.0 | 7.39 |
| 3.0 | 20.09 |

一般来说，在相同数据、相同 Tokenizer 和相同评估方式下，PPL 越低越好。

**特别注意：水果分类任务不需要使用 Perplexity。**

虽然数学上可以对分类交叉熵取指数，但那不是我们通常所说的语言模型 Perplexity。

未来训练 GPT 时，才会重点使用：

```python
import math

perplexity = math.exp(avg_token_loss)
```

这里的 `avg_token_loss` 必须是正确计算的平均 Token Loss，而不是随意拿一个分类 Loss 替代。

---

## 8\. Model Selection：如何选择最佳模型？

假设训练了 10 个 Epoch。

| Epoch | Train Loss | Val Loss | Val Accuracy |
|--------|--------|--------|--------|
| 1 | 0.85 | 0.88 | 75% |
| 2 | 0.62 | 0.65 | 82% |
| 3 | 0.42 | 0.45 | 88% |
| 4 | 0.30 | 0.32 | 92% |
| 5 | 0.21 | 0.28 | 94% |
| 6 | 0.14 | 0.30 | 93% |
| 7 | 0.10 | 0.36 | 91% |
| 8 | 0.07 | 0.43 | 89% |

你会发现：

Train Loss 一直下降，但是 Val Loss 在 Epoch 5 达到最低点后开始上升。

这说明模型可能开始过拟合。

所以，我们不能简单地认为最后一轮训练出来的模型就是最好的。

我们应该保存验证集表现最好的模型。

例如：

```python
best_val_loss = float("inf")

for epoch in range(num_epochs):

    train_loss = ...

    val_loss = ...

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
```

这里：

```python
float("inf")
```

表示正无穷大。

第一次得到任何正常的 Val Loss，都能满足：

```python
val_loss < best_val_loss
```

因此会保存第一轮模型。

之后只有 Val Loss 更低，才会覆盖之前的最佳模型。

**注意：最佳模型不一定是最后一个 Epoch 的模型。**

今天我们先用 Validation Loss 最低作为选择标准。对于类别严重不平衡的任务，也可以根据 Macro F1 选择模型，但标准应当在查看测试集结果之前确定。

## 9\. Test Evaluation：最终测试

训练结束后：

```python
checkpoint = torch.load(
    "best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

test_metrics = evaluate(model, test_loader)
```

这里要特别注意：

测试集只在模型选择完成后使用。

不能看了 Test Accuracy 不满意，就反复调整模型再重新测试，然后把最后一次结果当作完全独立的测试结果。

---

# 第四步：代码实践

现在进入真正的代码部分。

先不急着写完整项目，我们先掌握三个核心函数。

## 1\. 数据集划分：random\_split()

先看 PyTorch 提供的工具：

```python
from torch.utils.data import random_split
```

假设：

```python
dataset = TensorDataset(x, y)
```

现在划分：

```python
train_size = 210
val_size = 45
test_size = 45

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

这里重点理解：

```python
random_split(dataset, [210, 45, 45])
```

意思是：

把原来的 Dataset 随机分成三个 Subset。

注意：这里并没有复制三份数据，而是将原数据集的索引划分成三组。

然后分别创建 DataLoader：

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

为什么只有训练集设置：

```python
shuffle=True
```

因为训练集需要打乱数据顺序，减少模型对固定样本顺序的依赖。

验证集和测试集只负责评估，不需要打乱。

## 2\. 编写 evaluate() 函数

这是今天最重要的代码。

你之前的训练循环里，已经有：

```python
model.train()
```

今天我们需要认识：

```python
model.eval()
```

它们的区别：

| 方法 | 作用 |
|--------|--------|
| `model.train()` | 切换到训练模式 |
| `model.eval()` | 切换到评估模式 |

例如，Dropout 在训练时会随机屏蔽部分神经元，而在评估模式下会关闭这种随机屏蔽。

另外还要配合：

```python
with torch.no_grad():
```

它表示不计算梯度，减少评估时的内存占用。

注意：

- `model.eval()` 是切换模型行为。
- `torch.no_grad()` 是关闭梯度计算。

两者不是一回事。

### 2.1 先实现最基础的评估

```python
def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch_x, batch_y in loader:

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(batch_y)

            total_correct += (
                predictions == batch_y
            ).sum().item()

            total_samples += len(batch_y)

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy
```

这段代码与你之前的训练循环非常相似。

但是注意一个关键区别：

**评估时没有 `loss.backward()`，也没有 `optimizer.step()`。**

因为我们只是检查模型表现，不更新模型参数。

### 2.2 在此基础上增加混淆矩阵

我们现在需要计算 Precision、Recall 和 F1。

先创建一个矩阵：

```python
num_classes = 3

confusion_matrix = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.long
)
```

它会得到：

```text
tensor([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
```

接下来：

```python
predictions = torch.argmax(logits, dim=1)
```

假设：

```python
batch_y = torch.tensor([0, 1, 2, 0])

predictions = torch.tensor([0, 2, 2, 1])
```

我们希望得到：

```text
真实类别 0 → 预测类别 0
真实类别 1 → 预测类别 2
真实类别 2 → 预测类别 2
真实类别 0 → 预测类别 1
```

可以通过：

```python
encoded = batch_y * num_classes + predictions
```

得到：

```text
[0, 5, 8, 1]
```

然后：

```python
batch_cm = torch.bincount(
    encoded,
    minlength=num_classes ** 2
).reshape(num_classes, num_classes)
```

就可以得到：

```text
tensor([
    [1, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])
```

这里一定要理解：

```python
batch_y * num_classes + predictions
```

相当于把二维矩阵中的行列位置编码成一个一维数字。

例如：

- 真实类别 0，预测类别 0：0 × 3 + 0 = 0
- 真实类别 0，预测类别 1：0 × 3 + 1 = 1
- 真实类别 1，预测类别 2：1 × 3 + 2 = 5

`torch.bincount()` 就是在统计每个数字出现了多少次。

最后：

```python
confusion_matrix += batch_cm
```

就能把每个 Batch 的混淆矩阵累计起来，得到整个数据集的混淆矩阵。

### 2.3 从混淆矩阵计算 Precision、Recall、F1

现在有了混淆矩阵：

```python
tp = confusion_matrix.diag().float()

actual = confusion_matrix.sum(dim=1).float()

predicted = confusion_matrix.sum(dim=0).float()
```

分别是什么？

- `diag()`：取矩阵对角线，也就是真正预测正确的数量。
- `sum(dim=1)`：每一行求和，得到每个类别真实样本数量。
- `sum(dim=0)`：每一列求和，得到模型预测为每个类别的数量。

于是：

```python
precision = tp / predicted.clamp(min=1)

recall = tp / actual.clamp(min=1)

f1 = 2 * precision * recall / (
    precision + recall
).clamp(min=1e-12)
```

这里的：

```python
clamp(min=1)
```

是为了避免分母为零。

如果模型一次都没有预测某个类别，那么该类别的 Precision 就无法直接计算。我们在这里将它设为 0，避免程序报错。

最后：

```python
macro_precision = precision.mean().item()

macro_recall = recall.mean().item()

macro_f1 = f1.mean().item()
```

这样就得到了多分类的宏平均指标。

## 3\. Best Model：保存验证集表现最好的模型

现在我们把模型选择加入训练循环。

首先：

```python
best_val_loss = float("inf")
```

然后：

```python
if val_loss < best_val_loss:

    best_val_loss = val_loss

    torch.save(
        model.state_dict(),
        "best_model.pth"
    )
```

这样，每当验证集 Loss 创造新低，就覆盖之前的最佳模型。

训练结束：

```python
best_state = torch.load(
    "best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(best_state)
```

现在模型已经恢复为验证集表现最好的那一轮，而不是最后一轮。

---

# 第五步：Day 23 完整代码

文件名：`evaluation.py`

这次我们将前面所有知识串起来。

保留你的：

- 水果数据生成方式。
- `FruitClassifier` 网络结构。
- SGD 优化器。
- DataLoader。
- 原有 Forward、Loss、Backward、Optimizer 流程。

新增：

- Train / Val / Test 划分。
- evaluate()。
- Confusion Matrix。
- Precision / Recall / F1。
- Best Model 保存。
- 最终 Test Evaluation。

> 接口在完整程序加载最佳权重后的注释处截断，未取得后续原文。本地 evaluation.py 为空，因此下面保留取得的教学代码，并补全最后的 Test Evaluation 与结果打印；补全部分已在代码中标注。最佳模型文件改存脚本旁。

```python
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split

# ============================================
# 1. 基础配置
# ============================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_classes = 3

num_epochs = 20

batch_size = 32

from pathlib import Path

best_model_path = str(Path(__file__).with_name("best_model.pth"))

print("当前设备:", device)

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
# 3. 创建 Dataset
# ============================================

dataset = TensorDataset(x, y)

print("样本总数:", len(dataset))

# ============================================
# 4. 划分 Train / Validation / Test
# ============================================

train_size = int(0.7 * len(dataset))

val_size = int(0.15 * len(dataset))

test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(

    dataset,

    [train_size, val_size, test_size],

    generator=torch.Generator().manual_seed(42)
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# ============================================
# 5. 创建三个 DataLoader
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ============================================
# 6. 定义分类模型
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
# 7. 创建模型、Loss、Optimizer
# ============================================

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

# ============================================
# 8. 训练一个 Epoch
# ============================================

def train_one_epoch():

    model.train()

    total_loss = 0

    total_correct = 0

    total_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. Forward
        logits = model(batch_x)

        # 3. Loss
        loss = criterion(logits, batch_y)

        # 4. Backward
        loss.backward()

        # 5. Optimizer
        optimizer.step()

        # 6. 统计 Loss
        batch_size_now = batch_x.size(0)

        total_loss += loss.item() * batch_size_now

        # 7. 统计 Accuracy
        predictions = torch.argmax(logits, dim=1)

        correct = (
            predictions == batch_y
        ).sum().item()

        total_correct += correct

        total_samples += batch_size_now

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy

# ============================================
# 9. 评估函数
# ============================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0

    total_samples = 0

    # 混淆矩阵
    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)

            batch_y = batch_y.to(device)

            # 1. Forward
            logits = model(batch_x)

            # 2. Loss
            loss = criterion(logits, batch_y)

            batch_size_now = batch_x.size(0)

            total_loss += loss.item() * batch_size_now

            total_samples += batch_size_now

            # 3. 预测类别
            predictions = torch.argmax(logits, dim=1)

            # 4. 构建当前 Batch 的混淆矩阵
            encoded = (
                batch_y.cpu() * num_classes
                + predictions.cpu()
            )

            batch_cm = torch.bincount(
                encoded,
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

            # 5. 累加混淆矩阵
            confusion_matrix += batch_cm

    # ========================================
    # 10. 计算评估指标
    # ========================================

    avg_loss = total_loss / total_samples

    # 对角线：每个类别预测正确的数量
    tp = confusion_matrix.diag().float()

    # 每行求和：每个类别实际的数量
    actual = confusion_matrix.sum(dim=1).float()

    # 每列求和：每个类别被预测的数量
    predicted = confusion_matrix.sum(dim=0).float()

    # Precision
    precision = tp / predicted.clamp(min=1)

    # Recall
    recall = tp / actual.clamp(min=1)

    # F1
    f1 = 2 * precision * recall / (
        precision + recall
    ).clamp(min=1e-12)

    # Accuracy
    accuracy = tp.sum() / confusion_matrix.sum()

    # Macro Average
    macro_precision = precision.mean()

    macro_recall = recall.mean()

    macro_f1 = f1.mean()

    # Weighted F1
    weighted_f1 = (
        f1 * actual
    ).sum() / actual.sum()

    return {

        "loss": avg_loss,

        "accuracy": accuracy.item(),

        "precision": macro_precision.item(),

        "recall": macro_recall.item(),

        "f1": macro_f1.item(),

        "weighted_f1": weighted_f1.item(),

        "class_precision": precision.tolist(),

        "class_recall": recall.tolist(),

        "class_f1": f1.tolist(),

        "confusion_matrix": confusion_matrix
    }

# ============================================
# 11. 训练与 Validation
# ============================================

best_val_loss = float("inf")

best_epoch = 0

class_names = ["苹果", "香蕉", "橙子"]

for epoch in range(num_epochs):

    # 1. Training
    train_loss, train_accuracy = train_one_epoch()

    # 2. Validation
    val_metrics = evaluate(model, val_loader)

    # 3. Learning Rate
    current_lr = optimizer.param_groups[0]["lr"]

    # 4. 更新 Scheduler
    scheduler.step()

    print(f"\n========== Epoch {epoch + 1} ==========")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%}"
    )

    print(
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.2%}"
    )

    print(
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.6f}")

    # 5. 保存最佳模型
    if val_metrics["loss"] < best_val_loss:

        best_val_loss = val_metrics["loss"]

        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_loss": best_val_loss
            },
            best_model_path
        )

        print("发现更好的模型，已保存！")

# ============================================
# 12. 加载最佳模型
# ============================================

print("\n========== 加载最佳模型 ==========")

best_checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

print("最佳 Epoch:", best_checkpoint["epoch"])

print("最佳 Val Loss:", best_checkpoint["val_loss"])

# ============================================
# 13. 最终测试（归档补全，非取得的窗口原文）
# ============================================
# 仅在验证集选定并加载最佳权重之后评估测试集。
test_metrics = evaluate(model, test_loader)

print("\n========== 最终 Test Evaluation ==========")
for key in ("loss", "accuracy", "precision", "recall", "f1", "weighted_f1"):
    print(f"{key}: {test_metrics[key]:.6f}")

print("混淆矩阵（行是真实类别，列是预测类别）:")
print(test_metrics["confusion_matrix"])

for index, name in enumerate(class_names):
    print(
        f"{name}: Precision={test_metrics['class_precision'][index]:.4f} "
        f"Recall={test_metrics['class_recall'][index]:.4f} "
        f"F1={test_metrics['class_f1'][index]:.4f}"
    )
```

## 归档说明

保留可取得的全部概念讲解、公式、例子、逐步代码说明和教学程序。接口截断点之后的原文讲解尚未取得，不能把补全程序当作原文。测试集只在验证集选择并加载最佳模型之后使用，本例不对水果分类结果报告语言模型 Perplexity。来源：[《学习模型检查点保存》](https://chatgpt.com/c/6afeab87-2860-83e9-a278-bbd28e8b44dc)。
