# Day 9 · Forward & Loss

> 来源为《学习Day9前向与损失》，沿用原讲解顺序，保留推导、例子、反例与逐步代码说明。交互展示改为静态图表，公式适配 GitHub Markdown。原文件名在讲解中保留，本目录的 [code.py](code.py) 归档本地练习。

好，我们正式进入 **模块三：模型训练完整流程——Day 9：Forward & Loss（前向传播与损失计算）**。

这一天非常重要。因为从今天开始，我们要真正理解：**一个模型拿到数据之后，是怎样产生预测结果，又是怎样判断自己预测得好不好的。**

按照你要求的五步教学法来，今天不涉及 Backward 和 Optimizer，那是 Day 10 的内容。我们先把 Forward 和 Loss 吃透。

# Day 9：Forward & Loss

(Day 9 / 12) (模型训练完整流程) 预计 40–60 分钟

**今日学习目标**

1. 理解模型的前向传播（Forward）。
2. 理解 Logits 到底是什么。
3. 理解模型预测结果和真实标签的关系。
4. 掌握 CrossEntropyLoss。
5. 掌握 MSELoss。
6. 能够独立写出模型预测和损失计算代码。

---

# 第一步：用一个具体例子理解问题

假设我们现在要训练一个水果分类模型。

我们希望模型能够根据水果的特征，判断它属于哪一类：

- 苹果：类别 0
- 香蕉：类别 1
- 橙子：类别 2

我们给模型提供两个特征：

- 甜度
- 重量

假设有三条数据：

| 水果 | 甜度 | 重量 | 真实标签 |
|--------|--------|--------|--------|
| 苹果 | 0.8 | 0.6 | 0 |
| 香蕉 | 0.9 | 0.3 | 1 |
| 橙子 | 0.5 | 0.7 | 2 |

现在，我们把这三条数据交给神经网络。

模型需要做什么？

**第一件事：根据输入特征进行计算，产生预测结果。**

例如，模型对第一条数据（苹果）的预测结果可能是：

```python
[2.5, 0.8, 1.2]
```

这三个数字分别代表模型对三种水果的判断分数。

我们暂时不关心这些数字具体是怎么算出来的，只需要知道：

- 2.5：模型认为是苹果的分数。
- 0.8：模型认为是香蕉的分数。
- 1.2：模型认为是橙子的分数。

因为 2.5 最大，所以模型预测它是苹果。

```mermaid
flowchart LR
    X["甜度0.8、重量0.6"] --> F["神经网络 Forward"]
    F --> L["Logits 2.5、0.8、1.2"]
    L --> P["argmax 得到类别0，苹果"]
```

**第二件事：判断模型预测得好不好。**

我们已经知道第一条数据的真实答案是苹果（0）。

模型预测也是苹果（0）。

这次预测正确。

但是，如果模型输出：

```python
[0.2, 3.5, 1.1]
```

那么模型就预测成了香蕉。

这显然是错误的。

于是我们就产生了一个问题：

**能不能用一个数字，衡量模型预测得有多差？**

当然可以。

这个数字就是今天要学习的核心概念之一：Loss（损失）。

---

# 第二步：理解设计动机

我们先把今天的两个核心问题分开。

### 1\. 为什么需要 Forward？

因为神经网络本身只是一系列数学运算。

我们给它输入数据，它需要经过神经网络内部的权重、偏置、激活函数等计算，才能得到最终结果。

这个过程就是 Forward。

它解决的问题是：

**如何让模型根据输入数据产生预测结果？**

### 2\. 为什么需要 Loss？

模型能够产生预测结果，并不意味着它预测得正确。

我们必须把模型的预测结果与真实答案进行比较。

例如：

| 模型预测 | 真实答案 | 结果 |
|--------|--------|--------|
| 苹果 | 苹果 | 正确 |
| 香蕉 | 苹果 | 错误 |
| 橙子 | 橙子 | 正确 |

但是，仅仅知道对错还不够。

我们希望有一个数值来衡量模型的预测误差。

例如：

- Loss = 0.1：模型预测得比较好。
- Loss = 2.5：模型预测得比较差。

这里仅作直观举例，实际数值要结合具体损失函数理解。

这个数值就是 Loss。

**所以，Forward 负责产生预测，Loss 负责衡量预测质量。**

这是今天最重要的一句话。

---

# 第三步：专业概念与原理

接下来正式学习几个概念。

## 3.1 Forward：前向传播

专业定义：

前向传播是指输入数据经过神经网络中的各层计算，最终得到模型输出的过程。

例如：

```python
x = 输入数据

logits = model(x)
```

这里：

- `x`：输入数据。
- `model(x)`：将数据送入模型进行前向传播。
- `logits`：模型最终输出的原始分数。

注意一个重要的 PyTorch 语法：

```python
logits = model(x)
```

你之前学过 `nn.Module`。

这里调用 `model(x)`，实际上会触发模型的 `__call__()`，然后由它调用我们定义的 `forward()` 方法。

因此，你通常应该写：

```python
logits = model(x)
```

而不是：

```python
logits = model.forward(x)
```

这是 PyTorch 的标准使用方式。

## 3.2 Logits：原始预测分数

这是今天必须理解的概念。

刚才模型输出：

```python
[2.5, 0.8, 1.2]
```

这三个数字就是 Logits。

**Logits = 模型经过神经网络计算后输出的原始分数，尚未经过 Softmax 转换为概率。**

注意三个特点：

1. 可以是正数，也可以是负数。
2. 不要求加起来等于 1。
3. 数值越大，通常代表模型越倾向于对应类别。

例如：

```python
logits = torch.tensor([2.5, 0.8, 1.2])
```

如果想把它转换为概率，可以使用：

```python
probs = torch.softmax(logits, dim=0)
```

Softmax 会将原始分数转换为概率分布。

**Logits 与 Softmax 的关系**

| 类别 | 原始 Logit | Softmax 概率（约数） |
|---|---|---|
| 苹果 | 2.5 | 0.687 |
| 香蕉 | 0.8 | 0.126 |
| 橙子 | 1.2 | 0.187 |

静态表中的概率按本组 Logits 重新计算。

转换后，所有类别的概率之和为 1。

这里要区分两个东西：

- Logits：原始分数。
- Probabilities：概率。

不要混淆。

## 3.3 Prediction：预测结果

假设：

```python
logits = torch.tensor([
    [2.5, 0.8, 1.2],
    [0.3, 3.5, 1.1],
    [0.5, 0.7, 2.8]
])
```

这里有三条样本，每条样本有三个类别分数。

我们希望找到每一行最大的分数对应的类别。

使用：

```python
predictions = torch.argmax(logits, dim=1)
```

输出：

```python
tensor([0, 1, 2])
```

解释一下语法：

- `argmax()`：返回最大值所在的位置索引。
- `dim=1`：沿着第 1 个维度，也就是每一行的类别维度进行计算。

最终：

| 样本 | Logits | 预测类别 |
|--------|--------|--------|
| 1 | \[2.5, 0.8, 1.2\] | 0（苹果） |
| 2 | \[0.3, 3.5, 1.1\] | 1（香蕉） |
| 3 | \[0.5, 0.7, 2.8\] | 2（橙子） |

注意：`argmax` 得到的是类别索引，不是概率。

## 3.4 Label：真实标签

Label 就是真实答案。

例如：

```python
labels = torch.tensor([0, 1, 2])
```

它表示：

- 第一条样本真实类别是 0。
- 第二条样本真实类别是 1。
- 第三条样本真实类别是 2。

我们现在有了：

```python
predictions = [0, 1, 2]

labels = [0, 1, 2]
```

两者逐条对应。

注意：预测结果和真实标签必须按照样本顺序一一对应。

## 3.5 Loss：损失函数

现在正式进入今天最重要的部分。

Loss 的作用：

**衡量模型预测结果与真实标签之间的差距。**

不同任务使用的损失函数不同。

今天掌握两个：

- CrossEntropyLoss：分类任务。
- MSELoss：回归任务。

### A. CrossEntropyLoss：交叉熵损失

适用场景：分类任务。

例如：

- 判断图片是猫还是狗。
- 判断邮件是垃圾邮件还是正常邮件。
- 判断水果属于苹果、香蕉还是橙子。

我们先看一个简单例子。

假设真实答案是苹果（类别 0）。

模型经过 Softmax 后得到：

| 类别 | 模型预测概率 |
|--------|--------|
| 苹果（0） | 0.8 |
| 香蕉（1） | 0.1 |
| 橙子（2） | 0.1 |

模型认为苹果的概率是 80%，说明它比较有信心。

如果模型变成：

| 类别 | 模型预测概率 |
|--------|--------|
| 苹果（0） | 0.1 |
| 香蕉（1） | 0.8 |
| 橙子（2） | 0.1 |

这次模型就非常确信地预测错了。

CrossEntropyLoss 就是用来衡量这种预测情况的。

数学上，对于单条样本：

$$
L=-\log(p_{\text{真实类别}})
$$

其中：

- $p$：模型预测的概率。
- $p_{\text{真实类别}}$：真实类别对应的预测概率。

例如：

| 真实类别概率 | Loss |
|--------|--------|
| 0.9 | 0.105 |
| 0.8 | 0.223 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |

记住：真实类别的预测概率越高，Loss 越小。

**PyTorch 中的使用方法：**

```python
criterion = nn.CrossEntropyLoss()

loss = criterion(logits, labels)
```

这里：

- `criterion`：损失函数对象。
- `logits`：模型输出的原始分数。
- `labels`：真实标签。
- `loss`：计算得到的损失。

特别注意一个非常重要的工程细节：

**CrossEntropyLoss 直接接收 Logits，不需要提前 Softmax！**

也就是说：

(正确)

```python
loss = criterion(logits, labels)
```

(错误用法)

```python
probs = torch.softmax(logits, dim=1)

loss = criterion(probs, labels)
```

为什么？

因为 PyTorch 的 `CrossEntropyLoss` 内部已经包含了 LogSoftmax 和负对数似然损失的计算。

你再提前 Softmax，就重复处理了。

另外，分类标签一般应该使用 `torch.long` 类型的类别索引：

```python
labels = torch.tensor([0, 1, 2], dtype=torch.long)
```

不是普通的浮点数标签。

### B. MSELoss：均方误差损失

适用场景：回归任务。

回归任务不是预测类别，而是预测一个具体数值。

例如：

- 预测房屋价格。
- 预测明天的气温。
- 预测商品销量。

假设：

模型预测房价：

```python
predictions = [100, 200, 300]
```

真实房价：

```python
targets = [110, 190, 310]
```

每条数据的误差：

```python
[-10, 10, -10]
```

MSE 会把误差平方，再计算平均值。

公式：

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

其中：

- $y_i$：真实值。
- $\hat y_i$：预测值。
- $N$：样本数量。

上述例子：

$$
MSE=\frac{100+100+100}{3}=100
$$

PyTorch 使用：

```python
criterion = nn.MSELoss()

loss = criterion(predictions, targets)
```

### CrossEntropyLoss 和 MSELoss 的区别

| 对比 | CrossEntropyLoss | MSELoss |
|--------|--------|--------|
| 主要用途 | 分类 | 回归 |
| 模型输出 | 各类别 Logits | 预测数值 |
| 真实标签 | 类别索引 | 实际数值 |
| 衡量方式 | 真实类别的概率 | 预测值与真实值的平方差 |

这两个必须区分清楚。

---

# 第四步：代码实践

理论已经讲完，现在我们开始写代码。

今天我们先完成一个非常简单的分类模型。

目标：输入水果特征，经过神经网络，得到 Logits，再计算 CrossEntropyLoss。

## 4.1 第一步：准备数据

```python
import torch
import torch.nn as nn

# 4条样本，每条样本有2个特征
x = torch.tensor([
    [0.8, 0.6],
    [0.9, 0.3],
    [0.5, 0.7],
    [0.7, 0.4]
], dtype=torch.float32)

# 对应的真实标签
labels = torch.tensor(
    [0, 1, 2, 1],
    dtype=torch.long
)
```

理解：

```python
x.shape
```

输出：

```python
torch.Size([4, 2])
```

为什么？

- 4：有4条样本。
- 2：每条样本有2个特征。

而：

```python
labels.shape
```

输出：

```python
torch.Size([4])
```

因为每条样本只有一个类别标签。

## 4.2 第二步：构建模型

```python
class FruitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):

        return self.network(x)
```

逐行理解：

**第一部分：**

```python
class FruitClassifier(nn.Module):
```

定义一个神经网络模型，继承 `nn.Module`。

**第二部分：**

```python
def __init__(self):
    super().__init__()
```

初始化模型。

`super().__init__()` 用于初始化父类 `nn.Module`，让 PyTorch 正确管理模型中的参数和模块。

**第三部分：**

```python
self.network = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)
```

这里构建了一个简单的神经网络：

| 网络位置 | 输入 Shape | 输出 Shape |
|---|---|---|
| Linear(2,8) | `[B,2]` | `[B,8]` |
| ReLU | `[B,8]` | `[B,8]` |
| Linear(8,3) | `[B,8]` | `[B,3]` |

注意：

```python
nn.Linear(2, 8)
```

代表输入特征维度是 2，输出特征维度是 8。

```python
nn.Linear(8, 3)
```

代表输入维度是 8，输出维度是 3。

为什么最后是 3？

因为我们有三个分类类别：苹果、香蕉、橙子。

**第四部分：**

```python
def forward(self, x):
    return self.network(x)
```

定义前向传播过程。

当执行：

```python
model(x)
```

PyTorch 就会调用这个 `forward()` 方法。

## 4.3 第三步：执行 Forward

```python
model = FruitClassifier()

logits = model(x)

print(logits)
print(logits.shape)
```

这里：

```python
logits = model(x)
```

就是将输入数据送入模型，执行前向传播。

输出形状：

```python
torch.Size([4, 3])
```

为什么？

- 4：四条样本。
- 3：每条样本输出三个类别分数。

你要建立一个非常重要的 Shape 直觉：

$$
[B,\ Input]\rightarrow[B,\ Classes]
$$

本例：

$$
[4,2]\rightarrow[4,3]
$$

## 4.4 第四步：得到预测类别

```python
predictions = torch.argmax(logits, dim=1)

print(predictions)
```

这里使用 `argmax()`，找出每条样本中最大的分数所在位置。

例如：

```python
logits = torch.tensor([
    [2.5, 0.8, 1.2],
    [0.3, 3.5, 1.1],
    [0.5, 0.7, 2.8],
    [1.0, 2.0, 0.5]
])
```

输出：

```python
tensor([0, 1, 2, 1])
```

你可以发现：

```python
predictions = [0, 1, 2, 1]

labels = [0, 1, 2, 1]
```

两者完全一致。

说明这四条样本都预测正确。

当然，真实模型刚刚随机初始化，还没有经过训练，所以预测不一定正确。

## 4.5 第五步：计算 Loss

这就是今天最核心的代码。

```python
criterion = nn.CrossEntropyLoss()

loss = criterion(logits, labels)

print(loss)
```

拆解：

```python
criterion = nn.CrossEntropyLoss()
```

创建一个交叉熵损失函数。

```python
loss = criterion(logits, labels)
```

把模型预测的 Logits 和真实标签交给损失函数，计算预测误差。

注意这里：

- `logits`：形状 `[4, 3]`
- `labels`：形状 `[4]`
- `loss`：一个标量，也就是一个数字。

为什么四条数据最后得到一个 Loss？

因为 `CrossEntropyLoss()` 默认使用 `reduction="mean"`，即先计算每条样本的损失，再求平均。

你也可以这样观察：

```python
criterion = nn.CrossEntropyLoss(
    reduction="none"
)

loss = criterion(logits, labels)

print(loss)
```

这样就会输出四条样本各自的 Loss，而不是平均值。

---

# 第五步：Day 9 完整代码

现在把前面的内容整合成一个完整的 Python 文件。

文件名称：

`forward_loss.py`

这份代码可以直接复制运行，不需要额外下载数据集，也不需要安装 Transformers。

```python

import torch
import torch.nn as nn

# =====================================
# 1. 准备数据
# =====================================

# 4条样本，每条样本包含2个特征
x = torch.tensor([
    [0.8, 0.6],
    [0.9, 0.3],
    [0.5, 0.7],
    [0.7, 0.4]
], dtype=torch.float32)

# 真实标签
# 0: 苹果
# 1: 香蕉
# 2: 橙子

labels = torch.tensor(
    [0, 1, 2, 1],
    dtype=torch.long
)

# =====================================
# 2. 定义模型
# =====================================

class FruitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):

        return self.network(x)

# =====================================
# 3. 创建模型
# =====================================

model = FruitClassifier()

# =====================================
# 4. Forward：前向传播
# =====================================

logits = model(x)

print("模型输出 Logits：")
print(logits)

print("Logits 的形状：")
print(logits.shape)

# =====================================
# 5. Prediction：得到预测类别
# =====================================

predictions = torch.argmax(
    logits,
    dim=1
)

print("\n模型预测类别：")
print(predictions)

# =====================================
# 6. Softmax：转换为概率
# =====================================

probs = torch.softmax(
    logits,
    dim=1
)

print("\n模型预测概率：")
print(probs)

# =====================================
# 7. Label：真实标签
# =====================================

print("\n真实标签：")
print(labels)

# =====================================
# 8. Loss：计算交叉熵损失
# =====================================

criterion = nn.CrossEntropyLoss()

loss = criterion(
    logits,
    labels
)

print("\nCrossEntropyLoss：")
print(loss.item())

# =====================================
# 9. 观察每条样本的 Loss
# =====================================

criterion_each = nn.CrossEntropyLoss(
    reduction="none"
)

individual_losses = criterion_each(
    logits,
    labels
)

print("\n每条样本的 Loss：")
print(individual_losses)

# =====================================
# 10. 简单统计预测准确率
# =====================================

correct = (predictions == labels).sum().item()

total = labels.size(0)

accuracy = correct / total

print("\n预测正确数量：", correct)
print("总样本数量：", total)
print("Accuracy：", accuracy)
```

## 完整代码的数据流

你运行代码的时候，要在脑子里形成下面这条完整的数据流：

1. **准备输入数据**

   x：\[4, 2\]，labels：\[4\]
2. **执行 Forward**

   model(x) → logits：\[4, 3\]
3. **得到预测结果**

   argmax(logits) → predictions：\[4\]
4. **计算预测概率**

   softmax(logits) → probs：\[4, 3\]
5. **准备真实标签**

   labels：\[4\]
6. **计算 Loss**

   CrossEntropyLoss(logits, labels) → 一个标量

注意：第 3、4 步主要是为了观察和解释模型输出。计算交叉熵时，直接使用 Logits，不需要先执行这两步。

---

# 今日知识巩固：必须掌握的内容

我把今天最容易混淆的地方整理成一张表。

| 概念 | 作用 | PyTorch 写法 |
|--------|--------|--------|
| Forward | 前向传播 | `model(x)` |
| Logits | 原始预测分数 | `logits` |
| Softmax | 转换成概率 | `torch.softmax()` |
| Prediction | 获取预测类别 | `torch.argmax()` |
| Label | 真实标签 | `labels` |
| CrossEntropyLoss | 分类损失 | `nn.CrossEntropyLoss()` |
| MSELoss | 回归损失 | `nn.MSELoss()` |

**还有三个工程细节必须记住：**

1. `CrossEntropyLoss` 输入原始 Logits，不要提前 Softmax。
2. 分类任务的标签通常是 `torch.long` 类型的类别索引。
3. `loss.item()` 只是把 Loss 转换成普通 Python 数字，方便打印，不代表模型已经更新参数。

今天我们还没有进行 `loss.backward()`，也没有使用 `optimizer.step()`。所以模型参数不会因为这段代码而更新。

这就是 Day 10 要解决的问题。

---

# 今日学习验收

在进入 Day 10 之前，我希望你能真正理解下面四个问题。

原窗口此处为四道交互检测题，题目和作答未提供可读取内容。

**今天的实际学习要求：**

1. 把完整代码 `forward_loss.py` 自己敲一遍，不要直接复制粘贴。
2. 运行代码，观察每个变量的 Shape。
3. 重点理解 `logits`、`predictions`、`labels`、`loss` 之间的关系。
4. 完成上面的四道检测题。

等你完成后，我们再进入 Day 10：Backward & Optimizer。

下一天我们要真正揭开一个核心问题：

**模型算出了 Loss，究竟是怎么知道自己哪里错了，又是怎么调整内部参数，让下一次预测得更好的？**

这就是梯度、反向传播和参数更新。今天先把 Forward 和 Loss 的基础打牢。

---

## 归档练习与原讲解的对应关系

本目录 code.py 原样来自本地 forward_loss.py，包含模型、输入/标签、Logits、argmax 和 CrossEntropyLoss。上文的完整教学程序还演示 Softmax、逐样本 Loss 和 Accuracy，可整段独立运行。本地练习使用随机初始化，具体 Logits 和 Loss 随初始化变化。
