# Day 1：Tensor 核心操作

> 按原学习窗口的讲解顺序整理，保留设计动机、例子、反例、推导和逐步代码说明，并收录相关追问。原窗口交互组件转为静态表达，公式适配 GitHub。完整可运行的归档程序见 [code.py](code.py)。

好！我们正式开始 **PyTorch & Deep Learning Training — Day 1：Tensor 核心操作**。

这次我会按照你已经熟悉的 Transformer 教学方式来带你学，但有一个重点需要特别强调：

**我们的目标不是让你看懂代码，而是让你具备自己写 PyTorch 工程代码的能力。**

所以今天不会只讲几个 API，而是要让你真正理解：

- Tensor 到底是什么，为什么深度学习要使用它？
- Tensor 的维度和 Shape 应该怎么理解？
- 如何创建、操作和修改 Tensor？
- 索引、切片到底怎么写？
- 广播机制为什么能让不同 Shape 的 Tensor 相互运算？
- 矩阵运算和普通乘法有什么区别？
- 数据类型为什么重要？

今天的最终产出是一个可以独立运行的 `tensor_basics.py`。



(今天学习) (预计 60–90 分钟) (Python + PyTorch)

## 第一部分：先理解 Tensor 到底是什么

### 1\. 具体例子引入

假设我们正在训练一个 AI 模型，需要处理下面这些数据。

有一个学生的成绩：

```python
score = [85, 90, 78]
```

三个学生的成绩：

```python
scores = [
    [85, 90, 78],
    [92, 88, 95],
    [76, 80, 85]
]
```

你会发现：

- 一个学生的成绩，是一组数字。
- 三个学生的成绩，是一个二维数据表。
- 如果再增加班级，就可能变成三维数据。

这些数据本质上都是数字，只不过组织方式不同。

那么问题来了：

如果我们希望计算所有学生的平均成绩、进行矩阵运算，甚至让 GPU 帮我们同时处理几万个学生的数据，应该怎么做？

Python 原生的 List 虽然能存储数据，但并不是专门为高性能数值计算设计的。

于是，PyTorch 引入了一个非常核心的数据结构：

**Tensor（张量）。**

### 2\. 理解设计动机

Tensor 可以理解为一个专门为数值计算设计的多维数组。

它有几个重要特点：

1. 可以存储大量数值。
2. 支持各种数学运算。
3. 可以在 CPU 和 GPU 上运行。
4. 可以与自动求导机制配合，计算神经网络的梯度。

你可以先这样理解：

| 数据结构 | 可以理解为 |
|--------|--------|
| Scalar | 一个数字 |
| Vector | 一维数组 |
| Matrix | 二维数组 |
| Tensor | 更一般的多维数组 |

**Tensor 的维度直观理解**


**0 维：标量**

`5`


**1 维：向量**

`[1, 2, 3]`


**2 维：矩阵**

`[[1,2],[3,4]]`


**3 维：三维张量**

多个矩阵组成

这里有一个非常重要的认识：

**Tensor 的维度（Dimension）描述的是数据有多少个轴，而 Shape 描述的是每个轴有多少个元素。**

我们马上通过代码理解。

### 3\. 专业概念与原理：创建 Tensor

首先导入 PyTorch：

```python
import torch
```

最基础的创建方式：

```python
# 0维 Tensor
a = torch.tensor(5)

# 1维 Tensor
b = torch.tensor([1, 2, 3])

# 2维 Tensor
c = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(a)
print(b)
print(c)
```

逐个理解：

```python
a = torch.tensor(5)
```

- `torch`：PyTorch 库。
- `tensor()`：创建 Tensor 的函数。
- `5`：传入的数据。

所以，`a` 是一个存储数字 5 的 0 维 Tensor。

再看：

```python
c = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

这里创建了一个二维 Tensor。

它有两行，每行三个数字。

因此：

```python
print(c.shape)
```

输出：

```text
torch.Size([2, 3])
```

这就是 Shape。

**Shape = 每个维度上的元素数量。**

### 4\. Dimension 和 Shape 的区别

这是今天必须掌握的知识。

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(x.ndim)
print(x.shape)
```

输出：

```text
2
torch.Size([2, 3])
```

解释：

- `ndim`：有几个维度。
- `shape`：每个维度分别有多少个元素。

再结合你之前学习 Transformer 的知识：

```python
x = torch.randn(2, 10, 512)
```

它的 Shape 是：

```text
[2, 10, 512]
```

分别代表：

| 维度 | 含义 |
|--------|--------|
| 第 0 维：2 | Batch，两个句子 |
| 第 1 维：10 | 每句话有 10 个 Token |
| 第 2 维：512 | 每个 Token 有 512 维特征 |

因此：

```python
x.ndim   # 3
x.shape  # [2, 10, 512]
```

这与你之前手写 Multi-Head Attention 时使用的 `[B,T,C]` 是同一套理解方式。

---

## 第二部分：Tensor 的创建方式

实际工程中，我们不会总是手动输入数字。很多时候需要创建全零矩阵、随机矩阵或者指定范围的数据。

PyTorch 提供了很多创建 Tensor 的函数。

### 1\. 常用创建函数

```python
import torch

# 全部为 0
a = torch.zeros(2, 3)

# 全部为 1
b = torch.ones(2, 3)

# 全部为指定数字
c = torch.full((2, 3), 7)

# 随机生成 0~1 之间的数字
d = torch.rand(2, 3)

# 随机生成符合标准正态分布的数字
e = torch.randn(2, 3)

# 生成连续整数
f = torch.arange(0, 10)

# 生成单位矩阵
g = torch.eye(3)
```

逐个解释：

| 函数 | 作用 | Shape 示例 |
|--------|--------|--------|
| `torch.zeros()` | 创建全零 Tensor | `[2,3]` |
| `torch.ones()` | 创建全一 Tensor | `[2,3]` |
| `torch.full()` | 创建指定数值的 Tensor | `[2,3]` |
| `torch.rand()` | 均匀随机数 | `[2,3]` |
| `torch.randn()` | 标准正态分布随机数 | `[2,3]` |
| `torch.arange()` | 生成连续数值 | `[10]` |
| `torch.eye()` | 创建单位矩阵 | `[3,3]` |

注意两个容易混淆的地方：

```python
torch.rand(2, 3)
```

括号里的 `2, 3` 表示 Shape。

而：

```python
torch.full((2, 3), 7)
```

这里的 `(2, 3)` 是一个 Python 元组，表示 Shape，`7` 是填充的数值。

### 2\. 查看 Tensor 的基本属性

工程中经常需要检查 Tensor 的信息。

```python
x = torch.randn(2, 3, 4)

print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.device)
print(x.numel())
```

输出大致为：

```text
torch.Size([2, 3, 4])
3
torch.float32
cpu
24
```

这几个属性要记牢：

| 属性 | 含义 |
|--------|--------|
| `shape` | Tensor 的形状 |
| `ndim` | Tensor 的维度数量 |
| `dtype` | 数据类型 |
| `device` | 当前存储设备 |
| `numel()` | Tensor 中总共有多少个元素 |

例如：

```python
x = torch.randn(2, 3, 4)
```

总元素数量：

$$
2\times3\times4=24
$$

所以 `x.numel()` 返回 24。

这里特别提醒：`shape`、`ndim`、`dtype`、`device` 是属性，直接访问；`numel()` 是方法，需要加括号。

---

## 第三部分：索引与切片 Indexing & Slicing

### 1\. 具体例子引入

假设我们有三个学生的成绩：

```python
scores = torch.tensor([
    [85, 90, 78],
    [92, 88, 95],
    [76, 80, 85]
])
```

对应关系：

| 学生 | 数学 | 英语 | 计算机 |
|--------|--------|--------|--------|
| 学生0 | 85 | 90 | 78 |
| 学生1 | 92 | 88 | 95 |
| 学生2 | 76 | 80 | 85 |

现在我们想：

- 取出学生0的全部成绩。
- 取出学生1的数学成绩。
- 取出所有学生的数学成绩。
- 取出前两个学生的全部成绩。

这就需要索引和切片。

### 2\. 索引 Indexing

```python
# 取第一行
print(scores[0])

# 取第二行
print(scores[1])

# 取第二行第三列
print(scores[1, 2])

# 取第一行第二列
print(scores[0, 1])
```

输出：

```text
tensor([85, 90, 78])
tensor([92, 88, 95])
tensor(95)
tensor(90)
```

这里要理解：

**Python 和 PyTorch 的索引默认从 0 开始。**

对于二维 Tensor：

```python
scores[行索引, 列索引]
```

例如：

```python
scores[1, 2]
```

就是第二行、第三列，结果是 95。

### 3\. 切片 Slicing

切片的基本语法：

```python
tensor[start:end]
```

表示从 `start` 开始，取到 `end` 之前，不包含 `end`。

例如：

```python
# 取前两行
print(scores[0:2])

# 取所有行的第一列
print(scores[:, 0])

# 取所有行的前两列
print(scores[:, 0:2])

# 取前两行的前两列
print(scores[0:2, 0:2])
```

重点解释这个符号：

```python
:
```

它表示这一维全部取出。

因此：

```python
scores[:, 0]
```

意思是：

- 行：全部。
- 列：只取第 0 列。

输出：

```text
tensor([85, 92, 76])
```

这类操作在深度学习中非常常见。

例如之前 Mini-GPT 的生成代码：

```python
logits[:, -1, :]
```

你现在应该能理解：

- `:`：取全部 Batch。
- `-1`：取最后一个 Token。
- `:`：取这个 Token 的全部词表预测分数。

### 4\. 修改 Tensor 中的元素

```python
x = torch.tensor([1, 2, 3, 4])

x[0] = 100

print(x)
```

输出：

```text
tensor([100, 2, 3, 4])
```

也可以批量修改：

```python
x[1:3] = 0
```

结果：

```text
tensor([100, 0, 0, 4])
```

这里的 `=` 是直接修改原 Tensor 对应位置的值。

---

## 第四部分：广播机制 Broadcasting

(Day 1 核心重点)

这部分非常重要，尤其是你后续学习神经网络、Attention、Batch 计算时会频繁遇到。

### 1\. 具体例子引入

假设三个学生分别考了三门课：

```python
scores = torch.tensor([
    [80, 90, 70],
    [85, 75, 95],
    [90, 80, 85]
])
```

现在学校规定，三门课分别加上：

```python
bonus = torch.tensor([5, 10, 3])
```

也就是：

- 数学加 5 分。
- 英语加 10 分。
- 计算机加 3 分。

如果使用普通 Python List，我们可能需要写循环，逐个学生、逐门课程计算。

但 PyTorch 可以直接：

```python
result = scores + bonus

print(result)
```

结果：

```text
tensor([
    [85, 100, 73],
    [90, 85, 98],
    [95, 90, 88]
])
```

你可能会疑惑：

`scores` 的 Shape 是 `[3,3]`。

`bonus` 的 Shape 是 `[3]`。

一个二维 Tensor，一个一维 Tensor，为什么可以直接相加？

这就是 Broadcasting。

### 2\. 理解设计动机

深度学习经常需要对大量数据执行相同的操作。

例如：

- 一个 Batch 中有 32 个样本。
- 每个样本有 512 个特征。
- 希望对每个样本的 512 个特征都加上同一组偏置。

如果每次都手动复制数据、写循环，会非常麻烦。

广播机制允许不同 Shape 的 Tensor 在满足规则时自动扩展，使它们能够进行运算。

### 3\. 专业概念与规则

PyTorch 的广播规则：

**从最后一个维度开始，逐维比较两个 Tensor 的 Shape。**

每个维度必须满足以下条件之一：

1. 两个维度大小相等。
2. 其中一个维度大小为 1。
3. 其中一个 Tensor 没有这个维度，此时可以视为在前面补 1。

否则不能广播。

看几个例子：

**例子一：可以广播**

Tensor A **\[3, 4\]**

Tensor B **\[4\]**

---

可以相加

B 在前面补 1，变成 `[1,4]`，再扩展到 `[3,4]`。

结果 Shape：\[3,4\]

**例子二：可以广播**

Tensor A **\[3, 1\]**

Tensor B **\[1, 4\]**

---

可以相加

A 扩展成 `[3,4]`，B 也扩展成 `[3,4]`。

结果 Shape：\[3,4\]

**例子三：不能广播**

Tensor A **\[2, 3\]**

Tensor B **\[2, 4\]**

---

不能相加

从最后一维比较：3 和 4 不相等，而且都不是 1。

### 4\. 自己运行验证

```python
import torch

a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

b = torch.tensor([10, 20, 30])

c = a + b

print(c)
print(c.shape)
```

你可以发现：

```text
a.shape = [2, 3]
b.shape = [3]
c.shape = [2, 3]
```

再看一个：

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)

c = a + b

print(c.shape)
```

结果：

```text
torch.Size([3, 4])
```

注意：广播是运算时的逻辑扩展，不意味着一定会真的复制一份完整的数据。

**工程能力要求：看到两个 Tensor 相加，必须能够判断它们能不能广播，以及结果 Shape 是什么。**

---

## 第五部分：Tensor 的数学运算

### 1\. 普通运算

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

结果：

```text
tensor([5, 7, 9])
tensor([-3, -3, -3])
tensor([4, 10, 18])
tensor([0.2500, 0.4000, 0.5000])
```

这里有一个非常重要的区别：

```python
a * b
```

是逐元素相乘，不是矩阵乘法。

例如：

$$
[1,2,3]\times[4,5,6]
=[4,10,18]
$$

### 2\. 矩阵乘法 Matrix Multiplication

还记得你之前学习 Transformer 时，Attention 中有一个核心公式：

$$
QK^T
$$

这里使用的就是矩阵乘法。

先看一个简单例子：

```python
a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = torch.tensor([
    [5, 6],
    [7, 8]
])

c = a @ b

print(c)
```

结果：

```text
tensor([
    [19, 22],
    [43, 50]
])
```

为什么？

矩阵乘法规则：

$$
[A,B]\times[B,C]=[A,C]
$$

中间的两个维度必须相等。

具体计算：

$$
\begin{aligned}
c_{11}&=1\times5+2\times7=19\\
c_{12}&=1\times6+2\times8=22\\
c_{21}&=3\times5+4\times7=43\\
c_{22}&=3\times6+4\times8=50
\end{aligned}
$$

### 3\. 三种常见矩阵运算写法

```python
a * b
```

逐元素相乘。

```python
a @ b
```

矩阵乘法。

```python
torch.matmul(a, b)
```

也是矩阵乘法。

在 PyTorch 中，`@` 是 `torch.matmul()` 的简洁写法。

再联系之前学过的 Transformer：

```python
Q.shape = [2, 8, 10, 64]
K.shape = [2, 8, 10, 64]
```

如果：

```python
scores = Q @ K.transpose(-2, -1)
```

那么：

```text
Q                 [2, 8, 10, 64]
K.transpose(...)  [2, 8, 64, 10]

scores            [2, 8, 10, 10]
```

这就是你之前学习的 Attention Score 的计算。

今天的 Tensor 知识实际上是在为后续训练流程打底。

---

## 第六部分：数据类型 Dtype

### 1\. 具体例子引入

假设我们存储一个人的年龄：

```python
age = 25
```

存储一个人的身高：

```python
height = 163.5
```

年龄通常用整数就可以，而身高需要小数。

计算机为了存储这些数字，会使用不同的数据类型。

PyTorch 也一样。

### 2\. 常见数据类型

| Dtype | 含义 | 典型用途 |
|--------|--------|--------|
| `torch.int64` | 64 位整数 | 索引、Token ID |
| `torch.int32` | 32 位整数 | 整数数据 |
| `torch.float32` | 32 位浮点数 | 常规模型训练 |
| `torch.float16` | 16 位浮点数 | 混合精度训练 |
| `torch.bfloat16` | 16 位浮点数 | 大模型训练 |
| `torch.bool` | 布尔值 | Mask、条件判断 |

特别记住：

- Token ID 通常是整数。
- 模型参数通常是浮点数。
- Attention Mask 经常涉及布尔值。

### 3\. 查看和转换数据类型

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])

print(a.dtype)
print(b.dtype)
```

通常输出：

```text
torch.int64
torch.float32
```

转换类型：

```python
a = a.float()

print(a.dtype)
```

输出：

```text
torch.float32
```

也可以：

```python
a = a.to(torch.float32)
```

注意：

```python
a.int()
a.float()
a.double()
a.bool()
```

都是常见的数据类型转换方式。

工程中为什么要注意 Dtype？

因为某些运算要求数据类型兼容。例如，模型权重是 Float32，而输入数据是整数，直接进行某些运算可能出现类型不匹配。

此外，不同精度还会影响显存占用和数值稳定性。这部分我们在后续训练技巧中深入学习。

---

# 第七部分：Day 1 完整代码实践

前面学的是知识点，现在把它们整合成一个完整的 Python 文件。

创建文件：

`tensor_basics.py`

下面是今天的完整练习代码。建议你自己敲一遍，不要直接复制粘贴。

```python

import torch

# ==================================
# 1. Tensor 创建
# ==================================

a = torch.tensor([1, 2, 3])
b = torch.zeros(2, 3)
c = torch.ones(2, 3)
d = torch.randn(2, 3)

print("a:", a)
print("b:", b)
print("d:", d)

# ==================================
# 2. Tensor 基本属性
# ==================================

x = torch.randn(2, 3, 4)

print("shape:", x.shape)
print("ndim:", x.ndim)
print("dtype:", x.dtype)
print("device:", x.device)
print("numel:", x.numel())

# ==================================
# 3. 索引与切片
# ==================================

scores = torch.tensor([
    [85, 90, 78],
    [92, 88, 95],
    [76, 80, 85]
])

print("第一行:", scores[0])
print("第二行第三列:", scores[1, 2])
print("所有学生的数学成绩:", scores[:, 0])
print("前两个学生:", scores[:2])

# ==================================
# 4. Tensor 修改
# ==================================

x = torch.tensor([1, 2, 3, 4])

x[0] = 100
x[1:3] = 0

print("修改后:", x)

# ==================================
# 5. 广播机制
# ==================================

a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

b = torch.tensor([10, 20, 30])

c = a + b

print("广播结果:", c)
print("广播后的Shape:", c.shape)

# ==================================
# 6. 数学运算
# ==================================

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print("加法:", a + b)
print("乘法:", a * b)

# ==================================
# 7. 矩阵乘法
# ==================================

a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = torch.tensor([
    [5, 6],
    [7, 8]
])

c = a @ b

print("矩阵乘法:", c)

# ==================================
# 8. 数据类型
# ==================================

a = torch.tensor([1, 2, 3])

print("原始类型:", a.dtype)

a = a.float()

print("转换后类型:", a.dtype)
```

建议你按以下顺序运行：

1. 先运行第 1、2 部分，熟悉 Tensor 创建和属性。
2. 再运行第 3、4 部分，熟悉索引和修改。
3. 重点研究第 5 部分，自己判断广播规则。
4. 最后运行矩阵运算和数据类型部分。

---

# 第八部分：独立练习（Day 1 验收）

(现在轮到你动手)

接下来我不直接给答案。你需要自己写代码，我会逐题检查，尤其检查你的 Shape 判断、代码写法和工程习惯。

**Day 1 独立练习**

原窗口交互题的题目和作答未提供可读取内容，下面保留原讲解中的验收目标。


## Day 1 最终验收标准

今天学完，你应该能够做到：

- [ ] 不看资料，独立创建不同 Shape 的 Tensor。
- [ ] 能准确解释 `ndim`、`shape`、`dtype`。
- [ ] 能够根据行列索引和切片需求写出代码。
- [ ] 能够判断两个 Tensor 是否满足广播条件。
- [ ] 能够区分逐元素乘法和矩阵乘法。
- [ ] 能够根据矩阵 Shape 判断乘法结果。
- [ ] 能够进行基本的数据类型转换。
- [ ] 能够独立写出 `tensor_basics.py`。

**我对今天的教学有一个要求：不要为了赶进度而跳过没搞懂的语法。** 你之前学习 Transformer 时，对 `[B,T,C]`、矩阵乘法、维度变换这些问题都是通过具体数字逐步理解的。今天也继续保持这种方式。

今天先不进入 Day 2。等你完成独立练习，并且能够自己写出核心代码，我们再进入 Autograd（自动求导）。

**先从练习1开始，自己动手写。** 不会的地方直接问我，我们逐个攻克。

---

## 归档代码的补充说明

[code.py](code.py) 保留本人 tensor_basics.py 练习，示例数值与原讲解可能不同。创建部分反复覆盖 d，最终 dtype/device/numel 对应 eye(5)。广播结果已赋给 result。

类型转换返回结果需要接收。练习末尾只写 a.float()，a 的类型仍保持原值。

```python
a = torch.tensor([1, 2])
b = a.float()
print(a.dtype)  # torch.int64
print(b.dtype)  # torch.float32
a = a.to(torch.float32)
```

int()/float()/double()/bool() 分别返回 int32/float32/float64/bool 结果。转换为 bool 时0为 False，非0为 True。
