# Day 6：DataLoader —— 批量数据加载

> 按原学习窗口的讲解顺序整理，保留设计动机、例子、反例、推导和逐步代码说明，并收录相关追问。原窗口交互组件转为静态表达，公式适配 GitHub。完整可运行的归档程序见 [code.py](code.py)。

今天继续 PyTorch 工业训练的模块二。

Day 5 我们学会了 `Dataset`：**根据索引取出一条样本。**

今天要解决的问题是：**模型训练时，如何一次获取多条样本，而不是自己一条条读取？**

今天学完，你需要掌握：

| 知识点 | 重点 |
|--------|--------|
| DataLoader | 它到底解决什么问题 |
| Batch / Batch Size | 为什么批量训练 |
| Shuffle | 为什么训练时打乱数据 |
| Sampler | 谁决定取哪些样本 |
| `num_workers` | 多进程加载数据 |
| `pin_memory` | 加速 CPU → GPU 数据传输 |
| `drop_last` | 如何处理不完整的 Batch |

最后给你完整、可运行的 `dataloader.py`。

---

## 一、具体例子：为什么需要 DataLoader？

回忆 Day 5，我们已经写好了 Dataset。

假设有 6 条文本：

```python
texts = [
    "I love AI",
    "I hate AI",
    "Python is great",
    "This is terrible",
    "Deep learning is amazing",
    "I dislike this"
]

labels = [1, 0, 1, 0, 1, 0]
```

通过 Dataset，我们可以这样取数据：

```python
dataset[0]  # 第1条
dataset[1]  # 第2条
dataset[2]  # 第3条
```

但现在假设模型每次需要 **2 条数据一起训练**。

我们就需要这样组织：

6 条样本，Batch Size = 2

模型依次处理 Batch 1 → Batch 2 → Batch 3

当然，我们可以自己写循环，每次取两条。

但是，随着数据越来越多，我们还需要处理：

- 每次训练前随机打乱数据。
- 自动将样本划分成 Batch。
- 多进程提前加载数据，减少模型等待。
- 自动整理每个 Batch 中的数据。

这就是 **DataLoader 的作用：管理数据如何被取出，并组织成一批一批的数据。**

## 二、理解设计动机：Dataset 和 DataLoader 有什么区别？

直接对比：

```python
dataset[0]
```

得到一条样本：

```python
{
    "text": "I love AI",
    "label": 1
}
```

而：

```python
loader = DataLoader(dataset, batch_size=2)
```

创建一个批量数据加载器。

当我们遍历它时：

```python
for batch in loader:
    print(batch)
```

第一批可能是：

```python
{
    "text": ["I love AI", "I hate AI"],
    "label": tensor([1, 0])
}
```

看出区别了吗？

**Dataset 负责一条样本怎么取；DataLoader 负责什么时候取哪些样本，以及怎么把多条样本组成一个 Batch。**

DataLoader 本身不负责决定文本标签是什么，它通过调用 Dataset 获取样本。

---

## 三、Batch 和 Batch Size

### 1\. 什么是 Batch？

假设我们有 1000 条训练数据。

如果每次只用一条训练，模型就需要依次处理 1000 个样本。

我们可以一次输入 32 条，让 GPU 通过并行计算提高效率。

这 32 条样本组成的集合，就叫一个 **Batch（批次）**。

`Batch Size = 32` 表示每次模型前向计算使用 32 条样本（最后一批可能不足 32 条）。

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=32
)
```

### 2\. 为什么批量训练？

回忆 Day 4 的训练流程：

```python
optimizer.zero_grad()

output = model(x)
loss = criterion(output, y)

loss.backward()
optimizer.step()
```

当 `x` 是一个 Batch 时：

1. 模型对整批样本进行前向计算。
2. 计算这一批的 Loss。
3. 计算梯度。
4. 更新一次参数。

例如 1000 条样本，Batch Size = 32：

- 每个完整 Batch 有 32 条样本。
- 最后一个 Batch 有 8 条。
- 共进行 32 个 Batch 的训练。

这里必须区分：

**Batch Size 决定一次处理多少条样本；不是数据集里一共有多少条样本。**

### 3. 用表格观察 Batch Size

| 样本总数 | Batch Size | 保留尾批的批次数 | 最后一批大小 |
|---|---|---|---|
| 10 | 1 | 10 | 1 |
| 10 | 2 | 5 | 2 |
| 10 | 3 | 4 | 1 |
| 10 | 4 | 3 | 2 |
| 10 | 5 | 2 | 5 |

---

## 四、Shuffle：为什么需要打乱数据？

假设训练集数据按下面的顺序排列：

```text
正面 正面 正面 正面
负面 负面 负面 负面
```

如果 `batch_size=2`，不打乱：

```text
Batch 1 → 正面 正面
Batch 2 → 正面 正面
Batch 3 → 负面 负面
Batch 4 → 负面 负面
```

模型连续看到同一类别的数据，可能导致参数更新受到数据排列顺序的明显影响。

我们希望数据能够随机混合：

```text
Batch 1 → 正面 负面
Batch 2 → 负面 正面
Batch 3 → 正面 负面
Batch 4 → 负面 正面
```

所以设置：

```python
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

注意：随机打乱不保证每个 Batch 都恰好包含不同类别。

通常：

- 训练集：`shuffle=True`
- 验证集 / 测试集：`shuffle=False`

`shuffle=True` 改变的是**读取样本的顺序**，不会改变原始 Dataset 中的数据，也不会打乱文本和标签的对应关系。

---

## 五、Sampler：谁决定取哪些样本？

这是一个需要理解、但今天不必过度深入的概念。

假设 Dataset 有 5 条数据：

```text
索引：0 1 2 3 4
```

DataLoader 需要决定读取顺序。

例如：

```text
0 → 1 → 2 → 3 → 4
```

或者：

```text
3 → 0 → 4 → 1 → 2
```

**Sampler 就是负责提供样本索引的机制。**

你可以简单理解：

```text
Sampler 决定索引
        ↓
Dataset 根据索引取样本
        ↓
DataLoader 收集样本
        ↓
Collator 整理 Batch
```

PyTorch 中常见两种 Sampler：

| Sampler | 作用 |
|--------|--------|
| `SequentialSampler` | 按顺序取样 |
| `RandomSampler` | 随机取样 |

对于普通的索引式 Dataset：

```python
DataLoader(dataset, shuffle=False)
```

通常采用顺序采样。

```python
DataLoader(dataset, shuffle=True)
```

通常采用随机采样。

你也可以手动指定：

```python
from torch.utils.data import RandomSampler

sampler = RandomSampler(dataset)

loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler
)
```

**注意：自己指定 `sampler` 时，不要同时设置 `shuffle=True`。**

因为它们都涉及样本索引的选择，DataLoader 不允许这样的参数组合。

---

## 六、num\_workers：为什么需要多进程加载？

假设模型在 GPU 上训练。

每次训练都需要从磁盘读取数据、进行预处理，然后送给模型。

如果数据加载太慢，就会发生：

```text
GPU 训练完成
     ↓
等待 CPU 读取下一批数据
     ↓
GPU 空闲
     ↓
数据准备完毕
     ↓
GPU 继续训练
```

这显然会浪费计算资源。

所以 PyTorch 提供：

```python
num_workers=4
```

表示使用 4 个工作进程并行加载数据。

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)
```

要注意：

- `num_workers=0`：由主进程加载数据。
- `num_workers=4`：由 4 个工作进程参与加载数据。
- 并不是越大越好，进程创建、内存和数据传输都有开销。
- 小数据集或 Windows 环境学习时，先设置 `0` 最方便。

**多进程主要加速的是数据读取和预处理，而不是模型本身的 GPU 计算。**

---

## 七、pin\_memory：为什么能加速 GPU 数据传输？

这个概念放到实际训练里理解。

模型训练时，我们通常会将数据从 CPU 传输到 GPU：

```python
x = x.to("cuda")
```

如果使用：

```python
pin_memory=True
```

DataLoader 会尝试将返回的 Tensor 放在锁页（Pinned）CPU 内存中。

这种内存可以帮助 CPU → CUDA GPU 的数据传输更高效，尤其适合配合：

```python
x = x.to("cuda", non_blocking=True)
```

但注意：

1. `pin_memory=True` 不会自动把数据搬到 GPU。
2. 它主要对 Tensor 有意义，普通字符串本身不会因此加速。
3. 在 CPU-only 训练中通常没有必要开启。
4. 是否真正提高速度，需要结合数据量和硬件进行测试。

你目前先记住：

**`pin_memory` 是数据传输优化，不是数据读取顺序控制。**

---

## 八、drop\_last：最后一个 Batch 不够怎么办？

例如：

```text
数据集：10 条
Batch Size：4
```

正常情况：

```text
Batch 1：4 条
Batch 2：4 条
Batch 3：2 条
```

最后一批只有 2 条。

如果：

```python
drop_last=False
```

会保留最后一批。

如果：

```python
drop_last=True
```

则舍弃最后不足 4 条的 Batch。

**10 条数据 · Batch Size = 4**

| drop_last | 实际批次数 | 批次大小 | 本轮样本数 |
|---|---|---|---|
| False | 3 | 4、4、2 | 10 |
| True | 2 | 4、4 | 8 |

通常：

- 训练：多数情况下 `drop_last=False` 即可，某些要求固定 Batch 大小的场景使用 `True`。
- 验证和测试：通常 `False`，保证所有样本都参与评估。

注意，`drop_last=True` 配合 `shuffle=True` 时，不代表每个 Epoch 都丢弃相同的样本。

---

# 九、代码实践：完整 dataloader.py

今天我们继续使用 Day 5 的文本分类 Dataset。

为了让你真正理解 `DataLoader` 的工作过程，我们先不使用 Tokenizer。

这样可以清晰观察：

1. Dataset 如何返回一条样本。
2. DataLoader 如何调用 Dataset。
3. Sampler 如何决定样本顺序。
4. DataLoader 如何形成 Batch。
5. 如何取出一个 Batch 的文本和标签。

由于文本长度不一样，今天先让 Batch 保留原始字符串。Day 7–8 再学习 Tokenizer 和动态 Padding。

```python

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

# ==========================================
# 1. 准备原始数据
# ==========================================

texts = [
    "I love machine learning",
    "This movie is boring",
    "Deep learning is amazing",
    "I dislike this product",
    "Python is very useful",
    "This experience is terrible",
    "AI is the future",
    "I hate waiting"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]

# ==========================================
# 2. 定义 Dataset（复习 Day 5）
# ==========================================

class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):

        if len(texts) != len(labels):
            raise ValueError("文本与标签数量不一致")

        self.texts = texts
        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        return {
            "text": self.texts[index],
            "label": self.labels[index]
        }

# ==========================================
# 3. 创建 Dataset
# ==========================================

dataset = TextClassificationDataset(texts, labels)

print("数据集大小:", len(dataset))

# ==========================================
# 4. 创建 DataLoader
# ==========================================

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

# ==========================================
# 5. 遍历每个 Batch
# ==========================================

print("\n===== 顺序加载 =====")

for batch_index, batch in enumerate(loader):

    print("\nBatch:", batch_index)

    batch_texts = batch["text"]
    batch_labels = batch["label"]

    print("文本:", batch_texts)
    print("标签:", batch_labels)
    print("Batch大小:", len(batch_texts))

# ==========================================
# 6. 开启 Shuffle
# ==========================================

shuffle_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    num_workers=0
)

print("\n===== 随机加载 =====")

for batch_index, batch in enumerate(shuffle_loader):

    print("\nBatch:", batch_index)
    print("文本:", batch["text"])
    print("标签:", batch["label"])

# ==========================================
# 7. 使用 Sampler
# ==========================================

sampler = RandomSampler(dataset)

sampler_loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler,
    num_workers=0
)

print("\n===== RandomSampler =====")

for batch in sampler_loader:
    print(batch)

# ==========================================
# 8. 测试 drop_last
# ==========================================

drop_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    drop_last=True
)

print("\n===== drop_last=True =====")

for batch_index, batch in enumerate(drop_loader):

    print(
        "Batch:", batch_index,
        "样本数:", len(batch["text"])
    )

# ==========================================
# 9. 查看 Batch 总数
# ==========================================

print("\n数据集样本数:", len(dataset))
print("正常 Batch 数:", len(loader))
print("drop_last=True Batch 数:", len(drop_loader))
```

## 十、彻底理解这份代码的运行逻辑

这是今天最重要的工程部分。

我们只看这个核心代码：

```python
dataset = TextClassificationDataset(texts, labels)

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False
)

for batch in loader:
    print(batch)
```

假设一共有 8 条数据。

**第 1 步：创建 Dataset**
`__init__(texts, labels)`

保存 8 条文本及标签

**第 2 步：创建 DataLoader**
`batch_size=3, shuffle=False`

配置加载方式，尚未把所有 Batch 取出来

**第 3 步：开始 for 循环**

DataLoader 开始根据索引获取样本

第 4 步：第一批索引

依次调用 dataset\[0\]、dataset\[1\]、dataset\[2\]

**第 5 步：整理成一个 Batch**

3 条文本组成列表，3 个标签组成 Tensor

**第 6 步：返回给训练循环**
`for batch in loader`

之后继续读取下一批索引 3、4、5

这里还有一个非常重要的细节。

为什么我们明明没有编写 Data Collator，却仍然能够得到 Batch？

因为 **DataLoader 默认已经内置了一个整理 Batch 的函数，叫 `default_collate`**。

例如，Dataset 返回：

```python
{"text": "I love AI", "label": 1}
{"text": "I hate AI", "label": 0}
```

默认整理后：

```python
{
    "text": ["I love AI", "I hate AI"],
    "label": tensor([1, 0])
}
```

它会把同名字段组合起来，并将普通数值标签转换为 Tensor。

但是，它还不能自动把不同长度的 Token ID 序列 Padding 到相同长度。这就是 Day 8 自定义 Data Collator 的学习重点。

---

# 十一、Day 6 总结

| 概念 | 一句话理解 |
|--------|--------|
| Dataset | 定义如何取一条样本 |
| DataLoader | 按规则取样本并组成 Batch |
| Batch | 一次交给模型处理的一组样本 |
| Batch Size | 每批最多多少条样本 |
| Shuffle | 是否随机打乱读取顺序 |
| Sampler | 决定取样本的索引顺序 |
| num\_workers | 用多少工作进程加载数据 |
| pin\_memory | 优化 CPU → GPU 的数据传输 |
| drop\_last | 是否舍弃最后不足一批的数据 |
| default\_collate | DataLoader 默认的 Batch 整理函数 |

**今天真正要记住的代码只有这一段：**

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for batch in loader:
    # 拿到一批样本
    ...
```

创建 `DataLoader` 是配置数据加载方式；遍历 `loader` 时，才会不断获取 Batch。

你在 Day 5 学习的 `__getitem__()`，正是在这里派上用场的。

下一天 **Day 7：Tokenization**，我们就开始解决一个更关键的问题：

> 现在 Batch 里还是 `"I love AI"` 这样的字符串，但神经网络只能计算数值 Tensor。如何把文本转换成 Token ID，并正确处理不同长度的句子？


---

# 补充讲解 · enumerate 与 Batch 字典逐步拆解

这里其实涉及两个知识点：**`enumerate()` 的语法**和 **Batch 的数据结构**。我们直接用例子讲清楚。

## 一、`for batch_index, batch in enumerate(loader)` 是什么意思？

先看一个普通例子：

```python
fruits = ["apple", "banana", "orange"]

for index, fruit in enumerate(fruits):
    print(index, fruit)
```

输出：

```text
0 apple
1 banana
2 orange
```

`enumerate()` 的作用就是：**遍历数据时，同时获取索引和数据本身。**

它依次产生：

```python
(0, "apple")
(1, "banana")
(2, "orange")
```

Python 会自动拆开每个二元组：

```python
index, fruit = (0, "apple")
```

因此，回到你的代码：

```python
for batch_index, batch in enumerate(loader):
```

- `loader`：不断提供一个个 Batch。
- `enumerate(loader)`：给每个 Batch 加上编号。
- `batch_index`：当前是第几个 Batch，从 0 开始。
- `batch`：当前 Batch 的具体数据。

等价于下面这种更容易理解的写法：

```python
batch_index = 0

for batch in loader:
    print(batch_index, batch)
    batch_index += 1
```

## 二、batch 内部到底长什么样？

假设我们的 Dataset 返回以下三条样本：

```python
dataset[0] = {"text": "I love AI", "label": 1}
dataset[1] = {"text": "I hate AI", "label": 0}
dataset[2] = {"text": "Python is great", "label": 1}
```

现在设置：

```python
loader = DataLoader(dataset, batch_size=2, shuffle=False)
```

DataLoader 取出前两条样本，并使用默认的 `collate_fn` 整理。

**Dataset 返回两条样本**

```python
{"text": "I love AI", "label": 1}
{"text": "I hate AI", "label": 0}
```

DataLoader 默认整理

**最终的 batch**

```python
{
    "text": ["I love AI", "I hate AI"],
    "label": tensor([1, 0])
}
```

**看到了吗？`batch` 本身就是一个字典（dict）。**

这个字典有两个键：

- `"text"`：对应一个文本列表。
- `"label"`：对应一个标签 Tensor。

因此，你完全可以这样取：

```python
batch_texts = batch["text"]
```

得到：

```python
["I love AI", "I hate AI"]
```

再执行：

```python
batch_labels = batch["label"]
```

得到：

```python
tensor([1, 0])
```

注意：`batch["text"]` 不是 DataLoader 特有的语法，而是 **Python 字典通过键获取值的普通语法**。

如果你在 Dataset 的 `__getitem__()` 中返回的是：

```python
return {
    "sentence": text,
    "target": label
}
```

那么 Batch 对应的键也会变成 `"sentence"` 和 `"target"`，就需要写：

```python
batch["sentence"]
batch["target"]
```

## 三、把整个 for 循环执行一遍

假设 3 条样本，`batch_size=2`，`shuffle=False`。

| 循环次数 | batch\_index | batch\["text"\] | batch\["label"\] |
|--------|--------|--------|--------|
| 第一次 | 0 | `["I love AI", "I hate AI"]` | `tensor([1, 0])` |
| 第二次 | 1 | `["Python is great"]` | `tensor([1])` |

因此：

```python
for batch_index, batch in enumerate(loader):

    print("\nBatch:", batch_index)

    batch_texts = batch["text"]
    batch_labels = batch["label"]

    print("文本:", batch_texts)
    print("标签:", batch_labels)
    print("Batch大小:", len(batch_texts))
```

第一次循环输出：

```text
Batch: 0
文本: ['I love AI', 'I hate AI']
标签: tensor([1, 0])
Batch大小: 2
```

第二次循环输出：

```text
Batch: 1
文本: ['Python is great']
标签: tensor([1])
Batch大小: 1
```

**最后你只要记住两个核心点：**

1. `enumerate(loader)`：每次提供 `(批次编号, 当前Batch)`。
2. `batch["text"]`：因为 Batch 是一个字典，所以可以通过键获取其中的数据。

这里的 Batch 字典结构来自 Dataset 的返回格式，并由 DataLoader 默认的数据整理函数转换得到。

---

## 与归档代码对应的说明

[code.py](code.py) 原样归档本地 dataloader.py。8条样本、batch_size=3 时，保留尾批得到3/3/2，丢弃尾批得到3/3。随机采样的具体顺序每次可能不同，检查时应核对样本覆盖与文本/标签配对。

当前 num_workers=0。Windows 使用多进程加载时，Dataset 类放在模块顶层，创建和遍历 DataLoader 的入口使用 if __name__ == "__main__": 保护。
