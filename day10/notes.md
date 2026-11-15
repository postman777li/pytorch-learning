# Day 10：Backward & Optimizer（反向传播与参数更新）

> 来源为《学习Day9前向与损失》，保留原讲解顺序、推导、例子、反例与逐步代码说明，交互展示改为静态图表，公式适配 GitHub。完整归档程序见 [code.py](code.py)。

(Day 10 / 12) (模块三：模型训练完整流程)

今天我们接着 Day 9 学习。

上一天你已经掌握：

```python
logits = model(x)                 # Forward
loss = criterion(logits, labels)  # Loss
```

但是，这里存在一个关键问题：

**模型已经知道自己预测错了，也计算出了 Loss，可它究竟怎样调整内部参数，让下一次预测得更准确？**

这就是 Day 10 要解决的问题。

今天我们依然按照你要求的五步学习法：

1. 具体例子引入。
2. 理解设计动机。
3. 专业概念与底层原理。
4. 从简单代码开始，逐步理解 PyTorch 实现。
5. 给出完整可运行的 `backward_optimizer.py`，供你独立敲写。

今天的核心是理解这五行代码：

```python
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

新增的重点是 `zero_grad()`、`backward()` 和 `step()`。

---

## 第一步：通过一个例子理解参数更新

先暂时不考虑复杂神经网络。

假设我们要训练一个模型，根据学习时间预测考试成绩。

模型非常简单：

$$
\hat y=wx
$$

其中：

- $x$：学习时间。
- $w$：模型需要学习的参数。
- $\hat y$：模型预测的成绩。

假设某个学生学习了 2 小时，真实成绩是 10 分。

但是模型一开始的参数是：

$$
w=3
$$

那么预测结果：

$$
\hat y=3\times2=6
$$

预测 6 分，真实 10 分。

显然模型预测错了。

我们使用 MSELoss：

$$
L=(\hat y-y)^2
$$

得到：

$$
L=(6-10)^2=16
$$

现在的问题来了：

**模型应该怎样修改 w，才能让 Loss 变小？**

很显然，当前预测值偏低，因此需要把 $w$ 调大。

但计算机不会像我们一样直接凭直觉判断。

它需要一个数学机制来确定：

- 参数应该增加还是减少？
- 参数应该改变多少？

这就引出了两个核心机制。

**Backward**

计算梯度

告诉模型应该往哪个方向调整参数

**Optimizer**

更新参数

根据梯度真正修改参数

你可以先记住：

**Backward 只负责计算梯度，不负责修改参数。Optimizer 才真正负责修改参数。**

---

## 第二步：理解设计动机

### 2.1 为什么需要梯度？

我们继续刚才的例子。

$$
L=(wx-y)^2
$$

将 $x=2$、$y=10$ 代入：

$$
L=(2w-10)^2
$$

我们希望知道：

当 $w$ 变化时，Loss 会怎样变化？

数学上可以求导：

$$
\frac{\partial L}{\partial w}=2(2w-10)\times2
$$

整理得到：

$$
\frac{\partial L}{\partial w}=4(2w-10)
$$

当 $w=3$：

$$
\frac{\partial L}{\partial w}=4(6-10)=-16
$$

因此，梯度是：

$$
\boxed{-16}
$$

这个负数是什么意思？

它告诉我们：**在当前位置，沿着 w 增大的方向移动一小步，Loss 会下降。**

也就是说，模型应该增大 $w$。

相反，如果梯度是正数，那么沿着 $w$ 减小的方向移动一小步，就可以降低 Loss。

### 2.2 为什么还需要 Optimizer？

有了梯度，模型仍然不会自动更新参数。

我们还要决定参数每次变化的幅度。

最基本的参数更新公式：

$$
\boxed{w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial L}{\partial w}}
$$

其中 $\eta$ 是学习率（Learning Rate）。

假设：

$$
\eta=0.1
$$

那么：

$$
w_{\text{new}}=3-0.1\times(-16)=4.6
$$

更新之后，再预测一次：

$$
\hat y=4.6\times2=9.2
$$

新的 Loss：

$$
L=(9.2-10)^2=0.64
$$

对比一下：

| 阶段 | w | 预测值 | Loss |
|---|---|---|---|
| 更新前 | 3 | 6 | 16 |
| 更新后 | 4.6 | 9.2 | 0.64 |

Loss 从 16 降低到 0.64。

这说明参数更新的方向和幅度在这一次是有效的。

不过你要注意：**不是每次更新都会让 Loss 下降。** 特别是学习率过大或者使用不同 Batch 时，Loss 可能短暂升高。

### 2.3 用不同学习率观察一次更新

初始 w=3，梯度为-16，更新后 w=3+16η，Loss=(2w-10)²。

| 学习率 η | 更新后的 w | 更新后的 Loss |
|---|---|---|
| 0 | 3 | 16 |
| 0.01 | 3.16 | 13.5424 |
| 0.10 | 4.6 | 0.64 |
| 0.125 | 5 | 0 |
| 0.25 | 7 | 16 |
| 0.50 | 11 | 144 |

这就是学习率为什么重要。它不仅决定更新速度，也影响训练是否稳定。

---

## 第三步：专业概念与底层原理

今天重点掌握五个概念。

### 3.1 Backward：反向传播

你之前在 Day 2 学过 Autograd，现在要把它真正用到模型训练中。

PyTorch 中：

```python
loss.backward()
```

它会沿着之前 Forward 创建的计算图，利用链式法则，计算 Loss 对各个需要梯度的叶子张量的梯度。

例如：

```python
w = torch.tensor(3.0, requires_grad=True)

x = torch.tensor(2.0)
y = torch.tensor(10.0)

prediction = w * x
loss = (prediction - y) ** 2

loss.backward()

print(w.grad)
```

输出：

```text
tensor(-16.)
```

你需要理解：

- `requires_grad=True`：要求 PyTorch 跟踪涉及这个张量的运算，以便计算梯度。
- `prediction = w * x`：前向传播。
- `loss = ...`：计算损失。
- `loss.backward()`：根据计算图进行反向求导。
- `w.grad`：存储计算出来的梯度。

**最重要的是：执行 `loss.backward()` 之后，w 仍然是 3.0。**

因为它只计算梯度，并没有修改参数。

### 3.2 梯度到底存在哪里？

在 PyTorch 中，模型的可训练参数通常是 `nn.Parameter`。

每一个可训练参数都有一个 `.grad` 属性。

例如：

```python
for name, param in model.named_parameters():
    print(name, param.grad)
```

这里有两个之前可能不太熟悉的语法。

**`model.named_parameters()`**

它返回模型中各个参数的名字和参数张量。

例如：

```text
network.0.weight
network.0.bias
network.2.weight
network.2.bias
```

**`for name, param in ...`**

这是 Python 的解包语法。

每次循环得到两个值：

```python
name, param
```

分别接收参数名称和参数对象。

和你之前学过的：

```python
for batch_index, batch in enumerate(loader):
```

是类似的思路。

### 3.3 Optimizer：优化器

PyTorch 中最简单的优化器是 SGD（随机梯度下降）。

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

解释：

- `torch.optim.SGD`：创建 SGD 优化器。
- `model.parameters()`：提供模型需要更新的参数。
- `lr=0.1`：学习率。

注意，SGD 只是优化器的一种。

以后还会学习：

- Adam
- AdamW

在现代大模型训练中，AdamW 是一种常见选择。

但今天先用 SGD，因为它最容易理解参数更新的原理。

### 3.4 optimizer.step()：真正更新参数

```python
optimizer.step()
```

这行代码才是真正执行参数更新。

对于没有动量和权重衰减的普通 SGD：

$$
\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla_\theta L
$$

其中：

- $\theta$：模型的参数。
- $\eta$：学习率。
- $\nabla_\theta L$：Loss 对参数的梯度。

你可以把 `optimizer.step()` 理解为：优化器读取参数的 `.grad`，然后根据自己的更新规则修改参数。

### 3.5 optimizer.zero\_grad()：为什么要清零梯度？

这个非常重要。

你之前学过，PyTorch 的梯度默认会累积。

我们看一个例子：

```python
w = torch.tensor(3.0, requires_grad=True)

loss = (w * 2 - 10) ** 2

loss.backward()
print(w.grad)   # -16

loss.backward()
print(w.grad)   # -32
```

这里第二次 `backward()` 对同一个计算图可能报错，因为 PyTorch 默认会释放反向传播所需的部分中间结果。

如果需要重复反向传播同一张图，必须指定 `retain_graph=True`，但普通训练不应该这样写。

我们换成正确的梯度累积示例：

```python
w = torch.tensor(3.0, requires_grad=True)

loss1 = (w * 2 - 10) ** 2
loss1.backward()

print(w.grad)  # -16

loss2 = (w * 2 - 10) ** 2
loss2.backward()

print(w.grad)  # -32
```

为什么会变成 -32？

因为第一次梯度是 -16，第二次也是 -16。

PyTorch 默认进行累加：

$$
-16+(-16)=-32
$$

这不是一个 Bug，而是一种有意设计。

因为在实际深度学习中，我们可能希望：

- 累积多个 Batch 的梯度。
- 然后再统一更新参数。

这就是 Gradient Accumulation。

但如果我们不希望累积，就需要清零：

```python
optimizer.zero_grad()
```

它会清除优化器所管理参数的旧梯度，通常设置为 `None`。

注意：`zero_grad()` 不会修改模型权重，它清理的是梯度。

---

## 第四步：代码实践——亲眼观察参数更新

现在我们写一份最小的可运行代码。

这份代码只做一件事：

**观察参数在 backward 前后、optimizer.step() 前后的变化。**

### 4.1 创建最简单的模型

```python
import torch
import torch.nn as nn

model = nn.Linear(1, 1, bias=False)
```

这里：

```python
nn.Linear(1, 1, bias=False)
```

代表：

- 输入维度：1
- 输出维度：1
- 不使用偏置项

因此模型实际计算的是：

$$
\hat y=wx
$$

为什么关闭偏置？

因为今天我们想先观察单个参数 $w$ 的变化，避免多余的参数干扰理解。

### 4.2 手动设置模型参数

```python
with torch.no_grad():
    model.weight.fill_(3.0)
```

这里有两个重要语法。

**`with torch.no_grad():`**

表示在这个代码块中，PyTorch 不记录这些运算的梯度计算图。

为什么需要它？

因为我们现在只是手动初始化模型参数，不是在进行需要求导的前向计算。

**`model.weight.fill_(3.0)`**

表示把模型权重全部填充成 3.0。

这里的下划线 `_` 是 PyTorch 中常见的命名约定，表示原地操作（in-place）。

也就是说：直接修改原来的 Tensor，而不是创建一个新的 Tensor 返回。

### 4.3 准备数据

```python
x = torch.tensor([[2.0]])
y = torch.tensor([[10.0]])
```

注意这里为什么是双层中括号：

```python
[[2.0]]
```

因为 `nn.Linear` 需要输入的最后一维与 `in_features` 对应。

这里：

```python
x.shape  # [1, 1]
```

表示：

- Batch Size = 1
- 每条样本包含 1 个特征

### 4.4 创建损失函数和优化器

```python
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

这里用 MSELoss，因为我们预测的是一个具体数字。

### 4.5 执行一次完整更新

```python
# 1. 清零梯度
optimizer.zero_grad()

# 2. Forward
prediction = model(x)

# 3. Loss
loss = criterion(prediction, y)

# 4. Backward
loss.backward()

# 5. 更新参数
optimizer.step()
```

这就是一次最基本的模型训练迭代。

但是，你不能只背这五行。

我们要真正看清楚每一步发生了什么。

---

## 4.6 逐步观察参数变化

下面是一份完整的单步参数观察代码。

```python

import torch
import torch.nn as nn

# 创建模型
model = nn.Linear(1, 1, bias=False)

# 手动设置初始权重为 3
with torch.no_grad():
    model.weight.fill_(3.0)

# 准备数据
x = torch.tensor([[2.0]])
y = torch.tensor([[10.0]])

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# 初始参数
print("初始权重：", model.weight.item())

# 清零梯度
optimizer.zero_grad()

# 前向传播
prediction = model(x)
loss = criterion(prediction, y)

print("预测值：", prediction.item())
print("Loss：", loss.item())

# 反向传播
loss.backward()

print("Backward 后的梯度：", model.weight.grad.item())
print("Backward 后的权重：", model.weight.item())

# 参数更新
optimizer.step()

print("Optimizer Step 后的权重：", model.weight.item())

# 更新后重新进行一次预测
with torch.no_grad():
    new_prediction = model(x)
    new_loss = criterion(new_prediction, y)

print("更新后的预测值：", new_prediction.item())
print("更新后的 Loss：", new_loss.item())
```

理论上输出会接近：

```text
初始权重： 3.0

预测值： 6.0
Loss： 16.0

Backward 后的梯度： -16.0
Backward 后的权重： 3.0

Optimizer Step 后的权重： 4.6

更新后的预测值： 9.2
更新后的 Loss： 0.64
```

浮点数计算可能出现微小误差，因此你可能看到类似 `4.5999999` 的输出。

注意三件事：

**第一，Forward 不会修改模型参数。**

**第二，Backward 计算梯度，但不会修改权重。**

**第三，Optimizer Step 才会根据梯度修改权重。**

这三个区别必须彻底掌握。

---

# 第五步：Day 10 完整工程代码

文件名称：

`backward_optimizer.py`

这次我们在 Day 9 的基础上继续使用水果分类模型。

我们要观察一个真实的多层神经网络，在一次训练迭代中：

- 参数初始化是什么状态。
- 前向传播得到什么 Logits。
- Loss 是多少。
- Backward 后每层参数的梯度是什么。
- Optimizer Step 后，参数有没有变化。
- 更新后模型的 Loss 是否发生变化。

我们还会加入一些常见的工程写法，例如 `named_parameters()`、`detach()`、`clone()`、`torch.no_grad()`。

```python

import torch
import torch.nn as nn

# =====================================
# 1. 固定随机种子
# =====================================

torch.manual_seed(42)

# =====================================
# 2. 准备数据
# =====================================

x = torch.tensor([
    [0.8, 0.6],
    [0.9, 0.3],
    [0.5, 0.7],
    [0.7, 0.4]
], dtype=torch.float32)

labels = torch.tensor(
    [0, 1, 2, 1],
    dtype=torch.long
)

# =====================================
# 3. 定义模型
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
# 4. 初始化模型
# =====================================

model = FruitClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# =====================================
# 5. 保存更新前的参数
# =====================================

old_params = {}

for name, param in model.named_parameters():
    old_params[name] = param.detach().clone()

# =====================================
# 6. 清零旧梯度
# =====================================

optimizer.zero_grad()

# =====================================
# 7. Forward
# =====================================

logits = model(x)

predictions = torch.argmax(
    logits,
    dim=1
)

print("Logits：")
print(logits)

print("\n预测类别：")
print(predictions)

# =====================================
# 8. 计算 Loss
# =====================================

loss = criterion(logits, labels)

print("\n更新前 Loss：")
print(loss.item())

# =====================================
# 9. Backward
# =====================================

loss.backward()

print("\n各层参数的梯度：")

for name, param in model.named_parameters():
    print(f"\n参数名称：{name}")
    print("梯度形状：", param.grad.shape)
    print("梯度范数：", param.grad.norm().item())

# =====================================
# 10. Optimizer Step
# =====================================

optimizer.step()

# =====================================
# 11. 比较参数更新前后变化
# =====================================

print("\n参数更新情况：")

for name, param in model.named_parameters():

    old_value = old_params[name]
    new_value = param.detach()

    difference = new_value - old_value

    print(f"\n参数名称：{name}")
    print("参数变化范数：", difference.norm().item())

# =====================================
# 12. 重新计算更新后的 Loss
# =====================================

with torch.no_grad():

    new_logits = model(x)

    new_loss = criterion(
        new_logits,
        labels
    )

print("\n更新后的 Loss：")
print(new_loss.item())

# =====================================
# 13. 最终比较
# =====================================

print("\n========== 最终结果 ==========")

print("更新前 Loss：", loss.item())
print("更新后 Loss：", new_loss.item())

print("本次参数更新已完成")
```

## 完整代码中新增的工程知识点

这部分很重要。因为你的最终目标不只是看懂代码，而是能够自己编写工业训练代码。

### 1\. 为什么固定随机种子？

```python
torch.manual_seed(42)
```

神经网络的参数通常是随机初始化的。

如果不固定随机种子，每次创建模型，初始参数可能不同。

因此，每次运行得到的 Logits 和 Loss 也可能不同。

设置：

```python
torch.manual_seed(42)
```

可以使相同环境和相同执行流程下的随机初始化更容易复现。

注意：它不保证所有硬件、所有算子都完全确定。

### 2\. 为什么需要 detach().clone()？

```python
old_params[name] = param.detach().clone()
```

这行代码是今天很值得掌握的工程写法。

我们希望保存参数更新前的值，之后和更新后的参数进行比较。

先看：

```python
param.detach()
```

它返回一个与原参数共享底层数据、但不参与当前自动求导图的新 Tensor。

然后：

```python
.clone()
```

创建一份独立的数据副本。

因此：

```python
param.detach().clone()
```

得到的是一份独立的参数快照。

为什么不能直接写：

```python
old_params[name] = param
```

因为这只是保存了对同一个参数对象的引用。

当 `optimizer.step()` 修改参数时，`old_params[name]` 看到的也会是更新后的参数。

这样你就没办法比较更新前和更新后的数值了。

### 3\. 为什么使用字典保存参数？

```python
old_params = {}
```

这是一个空字典。

然后：

```python
for name, param in model.named_parameters():
    old_params[name] = param.detach().clone()
```

例如：

```python
old_params = {
    "network.0.weight": tensor(...),
    "network.0.bias": tensor(...),
    "network.2.weight": tensor(...),
    "network.2.bias": tensor(...)
}
```

这样我们就可以根据参数名称找到对应的旧参数：

```python
old_value = old_params[name]
```

这是很常见的 Python 工程写法。

### 4\. 为什么查看梯度范数？

```python
param.grad.norm().item()
```

其中：

```python
param.grad
```

代表当前参数的梯度张量。

```python
.norm()
```

默认计算向量的 L2 范数；对矩阵等张量，默认计算所有元素平方和开根号得到的范数。

你可以简单理解为：用一个数字概括这个张量中梯度的整体大小。

例如：

```python
gradient = torch.tensor([3.0, 4.0])

print(gradient.norm())
```

结果：

```text
tensor(5.)
```

因为：

$$
\sqrt{3^2+4^2}=5
$$

在实际训练中，梯度范数经常用来观察：

- 梯度是否过大。
- 梯度是否接近于零。
- 是否需要 Gradient Clipping。

这些内容会在后面的训练技巧模块继续深入。

### 5\. 为什么计算参数变化范数？

```python
difference = new_value - old_value

print(difference.norm().item())
```

这个数值的作用是：

**确认参数到底有没有被更新，以及整体变化量有多大。**

如果参数变化范数大于 0，说明参数发生了变化。

不过，参数变化范数为 0 并不一定代表代码出错，也可能是当前梯度为零、参数被冻结，或者更新量小到无法在当前浮点精度下表示。

### 6\. 为什么最后重新计算 Loss 要使用 no\_grad()？

```python
with torch.no_grad():

    new_logits = model(x)

    new_loss = criterion(new_logits, labels)
```

因为我们只是想检查参数更新后的预测表现。

暂时不需要计算新的梯度。

所以使用：

```python
torch.no_grad()
```

可以避免为这次检查创建自动求导计算图，从而减少不必要的内存和计算开销。

注意，`no_grad()` 并不等于 `model.eval()`。

前者控制自动求导图的记录；后者切换 Dropout、BatchNorm 等模块的运行行为。这两个概念作用不同，Day 12 会进一步学习。

---

# 第六部分：Day 10 的完整训练顺序

这一部分你以后几乎每天都会用到。

1. **optimizer.zero\_grad()**

   清理上一轮残留的梯度
2. **logits = model(x)**

   执行 Forward，得到预测结果
3. **loss = criterion(logits, labels)**

   计算模型预测与真实标签之间的损失
4. **loss.backward()**

   反向传播，计算各个可训练参数的梯度
5. **optimizer.step()**

   利用梯度更新模型参数

在普通训练中，以上顺序就构成了一次标准参数更新。

`zero_grad()` 也可以放在上一轮 `step()` 之后、下一轮前向传播之前；关键是正确控制梯度清零与累积。

## 一次训练到底是什么意思？

假设你有 1,000 条训练样本。

Batch Size = 100。

那么：

- 一次 Batch 处理 100 条数据。
- 一次 Epoch 要处理完 1,000 条数据。
- 一次 Epoch 包含 10 个 Batch。
- 如果每个 Batch 更新一次参数，那么一个 Epoch 中就会执行 10 次 `optimizer.step()`。

注意：**一次 Iteration 在常见训练语境中通常指处理一个 Batch 的一次循环，但采用梯度累积时，Iteration 数与参数更新次数可能不同。**

这一部分将在 Day 11 中系统学习。

---

# 第七部分：今日知识验收

先看你是否真正理解了 Day 10 的几个关键机制。

原窗口此处为五道交互检测题，题目和作答未提供可读取内容。

---

## Day 10 最终总结

今天我们真正完成了从“模型预测”到“模型参数更新”的学习。

| 步骤 | 核心代码 | 作用 |
|--------|--------|--------|
| 清零梯度 | `optimizer.zero_grad()` | 清除旧梯度 |
| Forward | `model(x)` | 计算模型输出 |
| Loss | `criterion(logits, labels)` | 衡量预测误差 |
| Backward | `loss.backward()` | 计算梯度 |
| Optimizer | `optimizer.step()` | 更新模型参数 |

最重要的是，你现在应该理解：

$$
\boxed{\text{Forward}\rightarrow\text{Loss}\rightarrow\text{Backward}\rightarrow\text{Update}}
$$

这是绝大多数基于梯度优化的深度学习模型训练流程的核心。

无论你之后训练：

- 简单 MLP
- CNN
- Transformer
- GPT
- 大语言模型

都离不开这些基本机制。

当然，实际大模型训练还会加入混合精度、梯度累积、分布式训练、学习率调度等机制，但核心逻辑并没有改变。

**Day 10 到这里结束。**

下一步 Day 11，我们要把今天学到的一次参数更新扩展成真正完整的训练循环：

`Dataset → DataLoader → Epoch → Batch → Forward → Loss → Backward → Optimizer Step`

届时你会亲自编写 `train_loop.py`，让模型自动训练多个 Epoch，观察 Loss 的下降过程，并真正理解 Batch、Iteration、Epoch 三者之间的区别。

---

## 归档练习与原讲解的对应关系

code.py 原样来自本地 backward_optimizer.py，保留固定种子42、参数快照、一次 SGD 更新、梯度与变化范数以及更新后 Loss 检查。上文同一计算图连续 backward 的片段是报错反例，应与随后重新前向的累积示例一起阅读。
