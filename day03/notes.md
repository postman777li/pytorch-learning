# Day 3：nn.Module 神经网络模型构建

> 按原学习窗口的讲解顺序整理，保留设计动机、例子、反例、推导和逐步代码说明，并收录相关追问。原窗口交互组件转为静态表达，公式适配 GitHub。完整可运行的归档程序见 [code.py](code.py)。

(PyTorch · Day 3) (概念 + 代码 + 工程实践)

今天我们要解决一个核心问题：

**Day 2 已经学会了计算梯度和更新参数，为什么还需要 `nn.Module`？**

学完今天，你应该能够不看资料，独立定义一个多层神经网络，并且理解每一行代码的作用。

## 今天的学习路线

| 顺序 | 内容 | 重点 |
|--------|--------|--------|
| 1 | 为什么需要 nn.Module | 理解设计动机 |
| 2 | `__init__` 与 `forward` | 模型如何定义与运行 |
| 3 | `nn.Linear` | 理解神经网络中的线性层 |
| 4 | `nn.Parameter` | 参数如何注册与训练 |
| 5 | `model.parameters()` | 优化器如何获得参数 |
| 6 | `nn.Sequential` | 简化网络结构 |
| 7 | `state_dict()` | 如何保存模型权重 |
| 8 | 完整代码实践 | 构建并训练多层神经网络 |

---

# 第一部分：为什么需要 nn.Module？

## 1\. 先回顾 Day 2

之前我们写过一个简单模型：

```python
import torch

x = torch.tensor(2.0)
y = torch.tensor(10.0)

w = torch.tensor(3.0, requires_grad=True)

pred = w * x
loss = (pred - y) ** 2

loss.backward()

with torch.no_grad():
    w -= 0.1 * w.grad
```

这里的模型只有一个参数 `w`。

但假设我们想构建一个神经网络：

**一个简单的多层神经网络**

示意图：每一层都有各自的权重和偏置。实际网络的神经元数量可以不同。

如果继续手动管理所有参数：

```python
W1 = torch.randn(3, 16, requires_grad=True)
b1 = torch.zeros(16, requires_grad=True)

W2 = torch.randn(16, 8, requires_grad=True)
b2 = torch.zeros(8, requires_grad=True)

W3 = torch.randn(8, 1, requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)
```

参数一多，就会出现几个问题：

- 每层参数都要自己管理。
- 优化器需要手动接收所有参数。
- 模型结构和训练代码混在一起。
- 保存、加载权重十分麻烦。
- 网络越来越复杂时难以复用。

## 2\. nn.Module 解决了什么？

PyTorch 提供 `nn.Module`，用来组织神经网络。

它能够：

1. 组织网络层及其计算逻辑。
2. 自动注册可训练参数。
3. 方便将参数交给优化器。
4. 提供保存和加载模型权重的机制。
5. 允许多个小模块组合成复杂网络。

**一句话：`nn.Module` 是 PyTorch 构建神经网络的基础类。**

你之前写的 `MultiHeadAttention`、`TransformerBlock`、`MiniGPT` 都是基于它实现的。

---

# 第二部分：`__init__` 和 forward 到底是什么？

先写一个最简单的网络：

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
```

现在我们逐行理解。

## 1\. class SimpleModel(nn.Module)

```python
class SimpleModel(nn.Module):
```

这是 Python 的类定义语法。

- `class`：定义一个类。
- `SimpleModel`：类名，可以自己取。
- `(nn.Module)`：表示继承 PyTorch 的 `nn.Module`。

继承后，这个类就具备了 PyTorch 模型的参数管理、模块管理、模式切换等能力。

## 2\. `__init__`：模型初始化

```python
def __init__(self):
    super().__init__()

    self.linear = nn.Linear(2, 1)
```

`__init__` 是 Python 类的初始化方法，创建实例时会自动执行。

例如：

```python
model = SimpleModel()
```

这时会执行 `__init__`，创建一个线性层。

### 为什么要写 super().\_\_init()？

```python
super().__init__()
```

是调用父类 `nn.Module` 的初始化方法。

这样 PyTorch 才能正确建立内部的参数和子模块注册机制。

### self 到底是什么？

```python
self.linear = nn.Linear(2, 1)
```

`self` 代表当前这个模型对象。

你可以把它理解为：

> 把这个线性层保存为模型自己的一个属性，以后在其他方法里也能使用。

比如：

```python
self.linear
```

是在模型内部访问这个线性层。

而外部可以：

```python
model.linear
```

访问同一个线性层。

注意区别：

```python
linear = nn.Linear(2, 1)
```

只是创建一个局部变量，不会作为模型属性被自动注册。

而：

```python
self.linear = nn.Linear(2, 1)
```

会把线性层注册为模型的子模块，它的参数也能通过 `model.parameters()` 找到。

**因此，应该把需要长期使用的网络层定义为 `self.xxx`。**

## 3\. forward：定义前向传播

```python
def forward(self, x):
    y = self.linear(x)
    return y
```

这段代码定义输入数据进入模型后，如何一步步计算输出。

例如：

```python
model = SimpleModel()

x = torch.tensor([[2.0, 3.0]])

y = model(x)

print(y)
```

这里的 Shape：

```text
x: [1, 2]
        ↓
Linear(2, 1)
        ↓
y: [1, 1]
```

为什么输入是 `[1,2]`？

因为：

- 1：一个样本。
- 2：每个样本两个特征。

`nn.Linear(2,1)` 会把两个输入特征转换为一个输出特征。

## 4\. 为什么调用 model(x)，而不是 model.forward(x)？

这是一个非常关键的 Python/PyTorch 机制。

```python
model(x)
```

本质上会触发 `nn.Module` 实现的调用流程，该流程再执行你定义的 `forward(x)`。

你可以把调用关系理解为：

```text
model(x)
    ↓
nn.Module.__call__()
    ↓
forward(x)
    ↓
返回结果
```

之所以不建议直接调用：

```python
model.forward(x)
```

是因为直接调用会绕开 `nn.Module` 的部分调用管理机制，例如前向 Hook。

所以工程中统一使用：

```python
output = model(x)
```

---

# 第三部分：nn.Linear 到底做了什么？

这是今天最重要的知识点之一。

## 1\. 从一个例子理解

假设我们用两个特征预测一个学生的成绩：

- `x1`：每天学习时间。
- `x2`：练习题数量。

我们希望模型学习：

$$
\hat y=w_1x_1+w_2x_2+b
$$

假设：

$$
x_1=2,\qquad x_2=3
$$

权重：

$$
w_1=4,\quad w_2=5,\quad b=1
$$

那么：

$$
\hat y=4\times2+5\times3+1=24
$$

这就是一个最简单的线性层。

## 2\. 使用 nn.Linear 实现

```python
linear = nn.Linear(2, 1)

x = torch.tensor([[2.0, 3.0]])

y = linear(x)
```

这里：

```python
nn.Linear(2, 1)
```

- `2`：输入特征维度。
- `1`：输出特征维度。

它会自动创建权重和偏置。

```python
print(linear.weight.shape)
print(linear.bias.shape)
```

输出：

```text
torch.Size([1, 2])
torch.Size([1])
```

注意：

**PyTorch 的 `nn.Linear(in_features, out_features)` 权重 Shape 是 `[out_features, in_features]`。**

对应的计算公式：

$$
\boxed{Y=XW^T+b}
$$

其中：

- 输入 `X`：`[B, in_features]`
- 权重 `W`：`[out_features, in_features]`
- 输出 `Y`：`[B, out_features]`

## 3\. 为什么要转置 W？

因为 PyTorch 把每个输出神经元对应的一组权重保存为一行。

例如：

```text
X: [1, 2]

W: [1, 2]
```

矩阵乘法需要：

```text
[1, 2] @ [2, 1] → [1, 1]
```

因此计算时使用：

```python
X @ W.T
```

但你不需要手动写这个运算。

```python
nn.Linear(2, 1)
```

已经封装好了。

## 4\. 手动设置参数，验证计算

现在我们不用随机权重，而是把刚刚例子中的参数填进去。

```python
import torch
import torch.nn as nn

linear = nn.Linear(2, 1)

with torch.no_grad():
    linear.weight.copy_(
        torch.tensor([[4.0, 5.0]])
    )

    linear.bias.fill_(1.0)

x = torch.tensor([[2.0, 3.0]])

y = linear(x)

print(y)
```

输出：

```text
tensor([[24.]], grad_fn=<AddmmBackward0>)
```

这里新增两个语法：

```python
linear.weight.copy_(...)
```

将指定 Tensor 的数值复制到已有的权重 Tensor 中。

```python
linear.bias.fill_(1.0)
```

将偏置 Tensor 中的所有元素填充为 1。

PyTorch 中很多以 `_` 结尾的方法属于原地操作，例如：

- `copy_()`
- `fill_()`
- `add_()`

为什么使用 `no_grad()`？

因为这是我们手动设置模型参数的操作，不希望它参与自动求导。

## 5\. 三维 Tensor 也能通过 Linear 吗？

当然可以！

例如：

```python
x = torch.randn(2, 10, 512)

linear = nn.Linear(512, 128)

y = linear(x)

print(y.shape)
```

输出：

```text
torch.Size([2, 10, 128])
```

解释：

```text
[B, T, C]
[2,10,512]
     ↓
Linear(512,128)
     ↓
[2,10,128]
```

`nn.Linear` 只对最后一个维度进行线性变换，前面的维度保持不变。

这就是你之前 Transformer 中：

```python
self.qkv = nn.Linear(512, 1536)
```

的运行原理。

---

# 第四部分：nn.Parameter 与模型参数注册

## 1\. 为什么普通 Tensor 不够？

假设我们自己定义一个参数：

```python
w = torch.tensor(3.0, requires_grad=True)
```

它可以计算梯度。

但是，如果我们把一个普通 Tensor 作为模型属性：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = torch.tensor(
            3.0, requires_grad=True
        )
```

`self.w` 虽然能自动求导，但不会被 PyTorch 自动注册为模型参数。

所以：

```python
model = Model()

print(list(model.parameters()))
```

输出：

```text
[]
```

这就产生一个问题：

如果优化器使用：

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1
)
```

它就找不到需要管理的参数。

## 2\. Parameter 如何解决？

使用：

```python
self.w = nn.Parameter(torch.tensor(3.0))
```

完整代码：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(
            torch.tensor(3.0)
        )

    def forward(self, x):
        return self.w * x

model = Model()

print(list(model.parameters()))
```

输出类似：

```text
[Parameter containing:
tensor(3., requires_grad=True)]
```

这说明模型现在能够识别并管理参数 `w`。

### Tensor 和 Parameter 的关键区别

|  | Tensor | nn.Parameter |
|--------|--------|--------|
| 存储数值 | 是 | 是 |
| 可以求导 | 设置 requires\_grad 后可以 | 默认可以 |
| 赋给 Module 属性后自动注册 | 否 | 是 |
| 自动出现在 parameters() 中 | 否 | 是 |

注意：`nn.Linear` 内部已经使用 `nn.Parameter` 管理权重和偏置。

因此：

```python
self.linear = nn.Linear(2, 1)
```

不需要你再手动把 `linear.weight` 包装成 `Parameter`。

---

# 第五部分：model.parameters() 是什么？

现在理解一个你之前见过很多次的代码：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)
```

为什么传入 `model.parameters()`？

因为它会返回模型中已经注册的参数，供优化器管理。

例如：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

查看参数：

```python
model = MyModel()

for name, param in model.named_parameters():
    print(name, param.shape)
```

输出：

```text
fc1.weight torch.Size([4, 2])
fc1.bias   torch.Size([4])
fc2.weight torch.Size([1, 4])
fc2.bias   torch.Size([1])
```

我们计算一下总参数量：

第一层：

$$
4\times2+4=12
$$

第二层：

$$
1\times4+1=5
$$

总参数量：

$$
12+5=17
$$

代码验证：

```python
total_params = sum(
    p.numel() for p in model.parameters()
)

print(total_params)  # 17
```

这里的 Python 语法：

```python
p.numel() for p in model.parameters()
```

是一个生成器表达式。

你可以把它理解为：

```python
total = 0

for p in model.parameters():
    total += p.numel()
```

两种写法作用相同。

**工程中 `named_parameters()` 用于检查参数名称和 Shape，`parameters()` 经常用于交给优化器。**

---

# 第六部分：nn.Sequential 是什么？

## 1\. 为什么需要 Sequential？

刚才我们定义网络：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

但如果我们的网络只是简单地：

```text
Linear → ReLU → Linear
```

其实可以用更简洁的方式。

## 2\. Sequential 写法

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

它会按照定义的顺序，把上一层的输出传给下一层。

例如：

```python
x = torch.randn(3, 2)

y = model(x)

print(y.shape)
```

输出：

```text
torch.Size([3, 1])
```

整个数据流：

**输入 X**

Shape: \[3,2\]

**nn.Linear(2,4)**

Shape: \[3,4\]

**nn.ReLU()**

Shape: \[3,4\]

**nn.Linear(4,1)**

Shape: \[3,1\]

### ReLU 是什么？

ReLU 是激活函数：

$$
ReLU(x)=\max(0,x)
$$

也就是：

- 输入大于 0：保持不变。
- 输入小于 0：变成 0。

例如：

```python
relu = nn.ReLU()

x = torch.tensor([-2.0, 0.0, 3.0])

print(relu(x))
```

输出：

```text
tensor([0., 0., 3.])
```

为什么需要它？

因为单纯堆叠多个线性层，本质上仍然可以合并成一个仿射变换。

加入 ReLU 这样的非线性激活函数，网络才能表达更复杂的非线性关系。

### Sequential 和自定义 Module 怎么选？

| 情况 | 推荐 |
|--------|--------|
| 简单顺序网络 | `nn.Sequential` |
| 复杂计算逻辑 | 自定义 `nn.Module` |
| 存在残差连接 | 自定义 `nn.Module` |
| Transformer Attention | 自定义 `nn.Module` |

你之前写的：

```python
x = x + self.attn(self.ln1(x))
```

就不适合直接用一个普通的 Sequential 表达，因为这里涉及残差连接。

---

# 第七部分：state\_dict() 是什么？

我们已经知道，模型包含大量参数。

但实际训练中，我们需要：

- 保存训练好的权重。
- 下次重新加载权重。
- 在不同环境中复用训练结果。

这就是 `state_dict()` 的作用。

## 1\. 查看模型状态

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

print(model.state_dict())
```

它会返回一个类似字典的结构，里面包含模型注册的参数及持久化缓冲区。

例如：

```text
0.weight
0.bias
2.weight
2.bias
```

为什么是 `0` 和 `2`？

因为：

```python
nn.Sequential(
    nn.Linear(2, 4),  # 第 0 层
    nn.ReLU(),         # 第 1 层
    nn.Linear(4, 1)    # 第 2 层
)
```

ReLU 没有需要保存的权重或偏置，所以没有 `1.weight`。

## 2\. 保存模型权重

```python
torch.save(
    model.state_dict(),
    "model_weights.pth"
)
```

解释：

- `torch.save()`：将对象序列化保存。
- `model.state_dict()`：获取模型的权重与缓冲区。
- `"model_weights.pth"`：保存文件路径。

## 3\. 加载模型权重

```python
new_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

state = torch.load(
    "model_weights.pth",
    map_location="cpu",
    weights_only=True
)

new_model.load_state_dict(state)
```

注意：

**加载权重之前，必须先创建与保存权重相匹配的模型结构。**

`state_dict()` 并不是把整个 Python 类定义保存了下来。

另外，`load_state_dict()` 会把权重加载到已有模型里，而不是返回一个新模型。

---

# 第八部分：完整工程实践 model\_basics.py

现在把今天所有知识组合起来。

我们的任务是：

**训练一个多层神经网络，学习两个输入特征与一个输出之间的关系。**

为了让结果可以验证，我们自己生成训练数据。

假设真实关系：

$$
y=2x_1+3x_2+1
$$

我们希望神经网络学会这个规律。

这里使用一个隐藏层：

```text
输入 [B,2]
   ↓
Linear(2,16)
   ↓
ReLU
   ↓
Linear(16,1)
   ↓
输出 [B,1]
```

## 完整代码

```python

import torch
import torch.nn as nn

# ===============================
# 1. 固定随机种子
# ===============================

torch.manual_seed(42)

# ===============================
# 2. 定义模型
# ===============================

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# 3. 创建模型
# ===============================

model = MyModel()

print("模型结构:")
print(model)

print("\n参数信息:")
for name, param in model.named_parameters():
    print(name, param.shape)

# ===============================
# 4. 构造训练数据
# ===============================

X = torch.rand(200, 2)

# 真实关系 y = 2*x1 + 3*x2 + 1
y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1

# ===============================
# 5. 定义损失函数和优化器
# ===============================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ===============================
# 6. 训练模型
# ===============================

for epoch in range(500):

    # 前向传播
    pred = model(X)

    # 计算损失
    loss = loss_fn(pred, y)

    # 清空旧梯度
    optimizer.zero_grad()

    # 计算梯度
    loss.backward()

    # 更新模型参数
    optimizer.step()

    # 每50轮输出一次
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1}, "
            f"Loss: {loss.item():.6f}"
        )

# ===============================
# 7. 测试模型
# ===============================

test_x = torch.tensor([[0.5, 0.8]])

with torch.no_grad():
    test_pred = model(test_x)

true_y = 2 * 0.5 + 3 * 0.8 + 1

print("\n测试预测:", test_pred.item())
print("真实答案:", true_y)

# ===============================
# 8. 保存权重
# ===============================

torch.save(
    model.state_dict(),
    "model_weights.pth"
)

print("模型权重保存完成")

# ===============================
# 9. 加载权重
# ===============================

new_model = MyModel()

state = torch.load(
    "model_weights.pth",
    map_location="cpu",
    weights_only=True
)

new_model.load_state_dict(state)

with torch.no_grad():
    loaded_pred = new_model(test_x)

print("加载后预测:", loaded_pred.item())
```

## 代码运行流程必须理解

**定义 MyModel**

告诉 PyTorch 模型长什么样

**创建 model = MyModel()**

执行 `__init__`，创建网络层与参数

**生成 X 和 y**

Shape: \[200,2\] → \[200,1\]

**model(X)**

触发 forward，计算预测值

**loss → backward → step**

计算误差、梯度并更新参数

**测试、保存和加载**

检验模型并复用权重

### 三个特别值得注意的地方

**① 为什么 y 使用 `X[:, 0:1]` 而不是 `X[:, 0]`？**

```python
X[:, 0].shape     # [200]
X[:, 0:1].shape   # [200,1]
```

我们希望 `y` 的 Shape 为 `[200,1]`，与模型输出保持一致。

这是很重要的工程习惯：避免因为广播机制而导致 Loss 被错误计算。

**② 为什么使用 `model.parameters()`？**

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

因为模型内部的 Linear 层已经自动注册了所有权重和偏置。

优化器读取这些参数，并在 `step()` 时更新它们。

**③ 为什么保存后还要加载？**

因为真正的模型工程不仅需要训练，还需要能够在之后恢复训练好的权重进行预测。

这里的测试输入：

```python
test_x = torch.tensor([[0.5, 0.8]])
```

理论真实值：

$$
y=2(0.5)+3(0.8)+1=4.4
$$

训练充分时预测值应该接近 4.4，但不保证恰好等于 4.4。

---

# 第九部分：独立练习

今天重点不是死记 API，而是能独立构建模型。

原讲解在此提供交互练习，下文保留可读取的讲解与验收目标。
原窗口的交互练习未提供可读取的题目和作答，按下方知识点自行核对。


---

## Day 3 最终总结

今天最需要记住的几个概念：

| 概念 | 一句话理解 |
|--------|--------|
| `nn.Module` | 神经网络的基础类 |
| `__init__` | 定义并初始化模型中的网络层 |
| `forward` | 定义输入数据如何计算得到输出 |
| `nn.Linear(a,b)` | 将最后一维从 a 变成 b |
| `nn.Parameter` | 能自动注册到 Module 中的参数 |
| `model.parameters()` | 获取已注册的模型参数 |
| `nn.Sequential` | 将多个模块按顺序串联 |
| `state_dict()` | 获取模型的参数和持久化缓冲区 |

**今天真正需要你具备的工程能力：**

能够独立写出：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

并且能解释：

- 为什么使用 `self`。
- 为什么要调用 `super().__init__()`。
- 为什么 `model(x)` 会执行 `forward()`。
- 为什么优化器能够自动找到模型参数。
- 每一层的输入和输出 Shape 是什么。

这才算真正完成 Day 3，而不是只是看懂了 `nn.Module` 的定义。

下一阶段 Day 4 会在这套代码基础上，学习 CPU/GPU 设备管理、`train()`、`eval()`、模型缓冲区以及训练可复现性。

---

## 与归档代码对应的说明

[code.py](code.py) 包含各部分短例子和原讲解的完整训练流程。权重路径整理为脚本旁的 model_weights.pth。最终网络 Linear(2,16) → ReLU → Linear(16,1) 共有65个参数，训练数据 Shape 为 [200,2]，标签为 [200,1]。

parameters() 返回全部注册参数，其中可能包含 requires_grad=False 的参数。只统计需要梯度的参数可写 sum(p.numel() for p in model.parameters() if p.requires_grad)。脚本运行与个人独立练习掌握情况分别记录。
