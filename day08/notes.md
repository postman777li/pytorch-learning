# Day 8：Data Collator —— 动态整理 Batch

> 来源为《学习Dataset数据集》的原讲解，保留原文顺序、例子和逐步代码说明。交互组件改为静态表格，公式适配 GitHub Markdown。可运行代码为本目录的 [code.py](code.py)，原始文件名在讲解中保留。

今天我们进入 PyTorch 数据处理模块的最后一天。

Day 5–7 已经学过：

- **Day 5 Dataset**：管理数据，按索引获取一条样本。
- **Day 6 DataLoader**：读取样本，组织成 Batch。
- **Day 7 Tokenizer**：将文本转换成 Token ID。

今天解决最后一个问题：

**不同长度的文本，如何在组成 Batch 时，自动整理成模型可以接收的统一格式？**

今天尤其重要的是理解 `DataLoader` 和 `Data Collator` 的协作关系。这是实际使用 Hugging Face Transformers 训练模型时非常常见的工程模式。

**Day 8 学习目标**

| 知识点 | 今天要掌握什么 |
|--------|--------|
| Data Collator | 它是什么、为什么需要 |
| 动态 Padding | 按当前 Batch 最长序列补齐 |
| `collate_fn` | DataLoader 如何调用 Collator |
| Batch 整理 | 多条样本如何组合成 Tensor |
| 标签处理 | 文本、标签如何分别组织 |
| 工程实践 | 独立理解并实现完整数据整理流程 |

---

## 一、具体例子：我们遇到了什么问题？

先回忆 Day 7。

假设 Tokenizer 已经把三条文本转换成了 Token ID：

```python
samples = [
    {
        "input_ids": [2, 3, 4],
        "label": 1
    },
    {
        "input_ids": [2, 3, 5, 6, 7],
        "label": 0
    },
    {
        "input_ids": [8, 9],
        "label": 1
    }
]
```

注意：

- 第一条文本有 3 个 Token。
- 第二条文本有 5 个 Token。
- 第三条文本有 2 个 Token。

现在我们希望把这三条数据组成一个 Batch。

直接转换 Tensor 会怎么样？

```python
torch.tensor([
    [2, 3, 4],
    [2, 3, 5, 6, 7],
    [8, 9]
])
```

会报错。

为什么？

因为普通二维 Tensor 要求每一行的长度一致。

但这里：

```text
第1条：[2, 3, 4]          长度3
第2条：[2, 3, 5, 6, 7]    长度5
第3条：[8, 9]             长度2
```

长度不一样。

所以我们需要先整理。

### 1\. 最简单的解决方法：Padding

我们可以把所有文本补齐到最长文本的长度，也就是 5。

假设 `[PAD] = 0`：

```python
[
    [2, 3, 4, 0, 0],
    [2, 3, 5, 6, 7],
    [8, 9, 0, 0, 0]
]
```

现在每条文本长度都是 5。

于是可以转换为：

```python
torch.tensor([
    [2, 3, 4, 0, 0],
    [2, 3, 5, 6, 7],
    [8, 9, 0, 0, 0]
])
```

(可以组成 Tensor)

但是问题来了：

**谁来负责这个 Padding？**

这就是 Data Collator 的工作。

## 二、设计动机：为什么不直接在 Tokenizer 中完成？

你可能会问：

Day 7 我们不是已经学过 Padding 了吗？

为什么还需要 Day 8？

关键在于：Padding 有两种常见策略。

### 1\. 固定长度 Padding

Day 7 学习的方式：

```python
tokenizer(
    text,
    padding="max_length",
    max_length=512
)
```

无论文本实际有多长，都补齐到 512。

例如：

```text
文本A：20个Token → 补到512
文本B：50个Token → 补到512
文本C：100个Token → 补到512
```

这样虽然统一了长度，但是可能浪费很多计算资源。

### 2\. 动态 Padding

Data Collator 通常采用另一种策略：

**先获取当前 Batch 的所有样本，再找出其中最长的序列，把其他样本补齐到这个长度。**

例如：

| Batch | 原序列长度 | 补齐后的统一长度 |
|---|---|---|
| Batch 1 | 3、5、2 | 5 |
| Batch 2 | 2、3 | 3 |

不同 Batch 可以有不同的序列长度。

这样做的好处：

- 减少不必要的 Padding。
- 减少无效计算。
- 降低显存占用。
- 提高训练效率。

注意：动态 Padding 是针对每一个 Batch 独立进行的，不是整个 Dataset 统一补齐。

## 三、专业概念：Data Collator 究竟是什么？

Data Collator 可以理解为一个**负责整理一批样本的函数或对象**。

它接收的通常是一个样本列表：

```python
[
    {"input_ids": [2, 3, 4], "label": 1},
    {"input_ids": [2, 3, 5, 6, 7], "label": 0},
    {"input_ids": [8, 9], "label": 1}
]
```

然后完成：

1. 找到当前 Batch 最长序列。
2. 对短序列进行 Padding。
3. 生成 Attention Mask。
4. 将数据转换成 Tensor。
5. 整理标签。

最后返回：

```python
{
    "input_ids": tensor([
        [2, 3, 4, 0, 0],
        [2, 3, 5, 6, 7],
        [8, 9, 0, 0, 0]
    ]),

    "attention_mask": tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ]),

    "labels": tensor([1, 0, 1])
}
```

这就是模型可以接收的一批输入。

### 特别理解：DataLoader 和 Data Collator 的关系

这里有一个很重要的知识点。

Day 6 我们学过：

```python
loader = DataLoader(
    dataset,
    batch_size=3
)
```

实际上 DataLoader 有一个参数：

```python
collate_fn
```

它决定如何把多条样本整理成一个 Batch。

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=3,
    collate_fn=my_collator
)
```

这意味着：

**DataLoader 负责取出样本，随后调用 `my_collator` 整理这些样本。**

具体流程：

**Dataset**

根据索引获取单条样本

**DataLoader**

收集 3 条样本

**collate\_fn**

调用自定义 Data Collator

**最终 Batch**

统一长度的 Tensor

还记得 Day 6 吗？

当时我们没有指定 `collate_fn`，为什么也能生成 Batch？

因为 DataLoader 默认使用 `default_collate`。

今天我们就是自己编写一个替代它的整理函数。

---

## 四、代码实践：从零编写 Data Collator

我们现在使用 Day 7 的知识。

假设 Tokenizer 已经完成了文本编码，每条样本已经是：

```python
{
    "input_ids": [...],
    "label": 1
}
```

今天暂时不重复实现 Tokenizer，而是专注于整理 Batch。

### 第一步：准备样本

```python
samples = [
    {
        "input_ids": [2, 3, 4],
        "label": 1
    },
    {
        "input_ids": [2, 3, 5, 6, 7],
        "label": 0
    },
    {
        "input_ids": [8, 9],
        "label": 1
    }
]
```

### 第二步：定义 Data Collator

```python
class MyDataCollator:

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
```

这里的 `__init__()` 用来保存 PAD 的 Token ID。

例如：

```python
collator = MyDataCollator(pad_token_id=0)
```

创建对象后：

```python
self.pad_token_id
```

就是 `0`。

### 第三步：编写 `__call__()`

这里会遇到一个新语法：

```python
def __call__(self, samples):
```

它的作用是让对象可以像函数一样被调用。

例如：

```python
class Person:

    def __call__(self):
        print("Hello")

person = Person()

person()
```

执行：

```text
Hello
```

为什么要用它？

因为我们希望 DataLoader 可以这样调用 Collator：

```python
collator(samples)
```

而不必写：

```python
collator.process(samples)
```

所以在 PyTorch 工程中，经常会看到把 Data Collator 写成带有 `__call__()` 的类。

### 第四步：找到最长序列

```python
max_length = max(
    len(sample["input_ids"])
    for sample in samples
)
```

这里有一个 Python 语法：

```python
max(表达式 for item in 列表)
```

它是生成器表达式。

相当于：

```python
lengths = []

for sample in samples:
    lengths.append(len(sample["input_ids"]))

max_length = max(lengths)
```

得到：

```python
max_length = 5
```

### 第五步：进行 Padding 和 Mask

```python
batch_input_ids = []
batch_attention_mask = []

for sample in samples:

    input_ids = sample["input_ids"]

    padding_length = max_length - len(input_ids)

    attention_mask = [1] * len(input_ids)

    input_ids = input_ids + [self.pad_token_id] * padding_length

    attention_mask = attention_mask + [0] * padding_length

    batch_input_ids.append(input_ids)

    batch_attention_mask.append(attention_mask)
```

这里的核心逻辑：

假设当前样本：

```python
input_ids = [2, 3, 4]
```

最长长度：

```python
max_length = 5
```

那么：

```python
padding_length = 5 - 3
```

结果：

```python
padding_length = 2
```

然后：

```python
input_ids + [0] * 2
```

得到：

```python
[2, 3, 4, 0, 0]
```

同时：

```python
attention_mask = [1, 1, 1, 0, 0]
```

### 第六步：整理标签

```python
labels = []

for sample in samples:
    labels.append(sample["label"])
```

得到：

```python
[1, 0, 1]
```

### 第七步：转换 Tensor 并返回

```python
return {
    "input_ids": torch.tensor(batch_input_ids),
    "attention_mask": torch.tensor(batch_attention_mask),
    "labels": torch.tensor(labels)
}
```

注意：

`input_ids` 和 `attention_mask` 是二维 Tensor。

`labels` 是一维 Tensor。

它们的形状分别为：

```text
input_ids:       [3, 5]
attention_mask:  [3, 5]
labels:          [3]
```

这就是模型训练需要的输入数据结构。

---

## 五、把 Data Collator 接入 DataLoader

前面我们只是单独编写了 Collator。

现在要把它真正接入 Day 5 和 Day 6 的代码。

先看这一句：

```python
loader = DataLoader(
    dataset,
    batch_size=3,
    collate_fn=collator
)
```

这里最重要的是：

```python
collate_fn=collator
```

意思是：DataLoader 每次收集好 3 条样本后，就把这 3 条样本交给 `collator`，让它负责整理。

注意，`collator` 是一个对象，但因为我们实现了 `__call__()`，所以它可以像函数一样被调用。

实际运行时，流程是：

```python
samples = [
    dataset[0],
    dataset[1],
    dataset[2]
]

batch = collator(samples)
```

DataLoader 内部就会完成类似的调用。

因此：

- `samples`：整理前的 3 条独立样本。
- `collator(samples)`：执行 Padding、Mask、标签整理。
- `batch`：整理好的模型输入。

## 六、完整代码：collator.py

这份代码是今天的最终工程实践，可以直接保存为 `collator.py` 运行。

它包含：

- Dataset：提供单条样本。
- Data Collator：动态 Padding、生成 Mask、整理标签。
- DataLoader：组织 Batch 并调用 Collator。
- 遍历 Batch：查看最终数据结构。

```python

import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 准备 Tokenizer 编码后的数据
# ==========================================

input_ids_list = [
    [2, 3, 4],
    [2, 3, 5, 6, 7],
    [8, 9],
    [2, 4, 5, 6],
    [8, 3]
]

labels = [1, 0, 1, 0, 1]

# ==========================================
# 2. 定义 Dataset
# ==========================================

class TextDataset(Dataset):

    def __init__(self, input_ids_list, labels):

        if len(input_ids_list) != len(labels):
            raise ValueError("数据和标签数量不一致")

        self.input_ids_list = input_ids_list
        self.labels = labels

    def __len__(self):

        return len(self.input_ids_list)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids_list[index],
            "label": self.labels[index]
        }

# ==========================================
# 3. 定义 Data Collator
# ==========================================

class MyDataCollator:

    def __init__(self, pad_token_id=0):

        self.pad_token_id = pad_token_id

    def __call__(self, samples):

        # 找出当前 Batch 中最长的序列长度
        max_length = max(
            len(sample["input_ids"])
            for sample in samples
        )

        # 创建空列表，保存整理后的数据
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        # 逐条整理样本
        for sample in samples:

            # 获取当前样本的 Token ID
            input_ids = sample["input_ids"]

            # 计算需要填充多少个 PAD
            padding_length = max_length - len(input_ids)

            # 真实 Token 对应 1
            attention_mask = [1] * len(input_ids)

            # Padding：在右侧补充 PAD
            input_ids = (
                input_ids
                + [self.pad_token_id] * padding_length
            )

            # Mask：在右侧补充 0
            attention_mask = (
                attention_mask
                + [0] * padding_length
            )

            # 保存整理后的数据
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

            # 保存标签
            batch_labels.append(sample["label"])

        # 将列表转换为 Tensor
        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long
            )
        }

# ==========================================
# 4. 创建 Dataset
# ==========================================

dataset = TextDataset(
    input_ids_list,
    labels
)

# ==========================================
# 5. 创建 Data Collator
# ==========================================

collator = MyDataCollator(
    pad_token_id=0
)

# ==========================================
# 6. 创建 DataLoader
# ==========================================

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=collator
)

# ==========================================
# 7. 遍历 Batch
# ==========================================

for batch_index, batch in enumerate(loader):

    print("\n========== Batch", batch_index, "==========")

    print("\nInput IDs:")
    print(batch["input_ids"])

    print("\nAttention Mask:")
    print(batch["attention_mask"])

    print("\nLabels:")
    print(batch["labels"])

    print("\n各部分 Shape:")
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
```

## 七、完整代码运行逻辑

我们用第一批数据走一遍。

原始 Dataset：

```python
dataset[0]
# {
#     "input_ids": [2, 3, 4],
#     "label": 1
# }

dataset[1]
# {
#     "input_ids": [2, 3, 5, 6, 7],
#     "label": 0
# }

dataset[2]
# {
#     "input_ids": [8, 9],
#     "label": 1
# }
```

DataLoader 设置：

```python
batch_size=3
```

因此它先取出这三条样本。

然后执行：

```python
collator(samples)
```

进入 `__call__()`。

第一步：找到最长序列，长度为 5。

第二步：分别进行 Padding。

```python
[
    [2, 3, 4, 0, 0],
    [2, 3, 5, 6, 7],
    [8, 9, 0, 0, 0]
]
```

第三步：生成 Attention Mask。

```python
[
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0]
]
```

第四步：整理标签。

```python
[1, 0, 1]
```

第五步：全部转换为 Tensor。

最终得到：

```text
input_ids:       torch.Size([3, 5])
attention_mask:  torch.Size([3, 5])
labels:          torch.Size([3])
```

第二批只有两条样本，它会重新计算自己的最长序列长度，而不是沿用第一批的长度。

这就是动态 Padding。

---

## 八、工业界实际怎么写？

实际使用 Hugging Face Transformers 时，很多时候不需要自己手写 Data Collator。

例如：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)
```

然后：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=data_collator
)
```

它会根据当前 Batch 的样本长度动态 Padding，并整理相应的输入字段。

注意：这里的 Dataset 必须返回 Tokenizer 能识别的字段，例如 `input_ids`，标签字段通常使用 `labels`；实际标签处理也要符合具体模型任务的要求。

你现在需要理解的是：**工业界不是不需要 Data Collator，而是很多时候使用已经封装好的 Data Collator。**

自己手写一次，是为了真正理解它内部在做什么。

---

## 九、Day 5–8 模块二最终验收：完整数据输入管线

现在我们把四天的知识串起来。

**原始文本**

"I love AI"

**Day 7 · Tokenizer**

文本 → Token ID

`[2, 3, 4]`

**Day 5 · Dataset**

封装样本，按索引获取

**Day 6 · DataLoader**

取出多个样本，组成 Batch

**Day 8 · Data Collator**

动态 Padding、Attention Mask、标签整理

**模型输入 Batch**

input\_ids / attention\_mask / labels

### 四天知识点总复习

| 模块 | 核心问题 | 关键代码 |
|--------|--------|--------|
| Day 5 Dataset | 一条样本怎么获取？ | `dataset[index]` |
| Day 6 DataLoader | 多条样本怎么组织？ | `DataLoader()` |
| Day 7 Tokenizer | 文本怎么变成数字？ | `tokenizer()` |
| Day 8 Data Collator | 不同长度怎么整理？ | `collate_fn` |

**Day 8 最重要的一句话：**

Data Collator 接收一批独立样本，负责把它们整理成模型能够处理的统一 Batch；DataLoader 通过 `collate_fn` 调用它。

到这里，模块二的数据处理与输入管线就完成了。

下一阶段可以进入模型训练工程实践，把这些数据真正送进 Transformer，完成 Forward、Loss、Backward 和 Optimizer 更新。
