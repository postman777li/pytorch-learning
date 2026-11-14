# Day 5 · Dataset 数据集

> 按原学习窗口的讲解顺序整理，保留设计动机、例子、反例、推导和逐步代码说明，并收录相关追问。原窗口交互组件转为静态表达，公式适配 GitHub。完整可运行的归档程序见 [code.py](code.py)。

我们正式开始 **PyTorch & Deep Learning Training — 模块二 Day 5：Dataset**。

今天我们不只是学会调用 `Dataset`，而是要真正理解它的设计思想，最后能够独立编写一个可以用于实际模型训练的文本分类 Dataset。

今天尤其注意 Python 面向对象编程的几个语法，因为它们会贯穿后面所有 PyTorch 工程代码。

**Day 5 · Dataset** (开始学习)

# 5 个核心知识点

预计学习时间：40–60 分钟

- Dataset 的作用与设计动机
- `__init__()`、`__len__()`、`__getitem__()`
- 样本（Sample）与标签（Label）
- Dataset 的底层调用逻辑
- 训练集、验证集划分

  ---

  **最终成果：** 独立编写 `dataset.py`，实现一个文本分类数据集。

# 一、具体例子引入：我们遇到了什么问题？

假设我们正在训练一个情感分类模型。

我们的任务是：判断一句英文评论是正面还是负面。

现在有以下数据：

| 文本 | 标签 |
|--------|--------|
| I love AI | 1 |
| This is terrible | 0 |
| AI is amazing | 1 |
| I hate this | 0 |
| Very good work | 1 |

这里我们约定：

- `1`：正面情感（Positive）
- `0`：负面情感（Negative）

现在假设我们有 10,000 条这样的数据。

我们需要训练模型，于是模型需要不断地读取这些数据。

最简单的办法是什么？

```python
texts = [
    "I love AI",
    "This is terrible",
    "AI is amazing",
    "I hate this",
    "Very good work"
]

labels = [1, 0, 1, 0, 1]
```

假设现在要获取第三条数据：

```python
text = texts[2]
label = labels[2]

print(text)
print(label)
```

输出：

```text
AI is amazing
1
```

看起来没有任何问题。

但是，随着数据规模越来越大，会出现一些问题：

1. 数据可能来自文件、数据库，而不是直接写在代码里。
2. 每条样本可能需要进行预处理。
3. 数据和标签需要保持对应关系。
4. 训练过程中需要频繁地按照索引获取数据。
5. 不同任务可能有不同的数据读取方式。

如果每次训练都自己编写一套数据读取代码，就会非常混乱。

**所以，我们需要一种统一的数据管理机制。**

这就是 Dataset 要解决的问题。

# 二、理解设计动机：为什么需要 Dataset？

我们先把问题抽象一下。

假设模型需要获取第 3 条数据。

我们希望有一种统一的方式：

```python
dataset[2]
```

不管数据存储在哪里，也不管数据读取过程中有多少操作，只要我们能够通过索引获取一条完整样本就可以。

例如：

```python
dataset[0]
```

得到：

```python
{
    "text": "I love AI",
    "label": 1
}
```

再执行：

```python
dataset[1]
```

得到：

```python
{
    "text": "This is terrible",
    "label": 0
}
```

这样，外部程序就不需要关心数据具体是怎么存储、怎么读取的。

它只需要知道两件事：

- 数据集一共有多少条数据？
- 给定一个索引，如何获取对应的数据？

这就是 Dataset 的核心设计思想。

**Dataset 的两个核心职责**

# 1\. 获取数据集长度

`len(dataset)`

告诉外部程序一共有多少条样本。

# 2\. 根据索引获取样本

`dataset[index]`

告诉外部程序如何取出指定的一条样本。

# 三、专业概念与原理：Dataset 究竟是什么？

PyTorch 提供了一个 Dataset 基类：

```python
from torch.utils.data import Dataset
```

我们可以通过继承它，创建自己的数据集。

这里涉及三个必须掌握的 Python 方法：

| 方法 | 作用 | 对应操作 |
|--------|--------|--------|
| `__init__()` | 初始化数据 | 创建 Dataset 对象时调用 |
| `__len__()` | 返回数据集长度 | `len(dataset)` |
| `__getitem__()` | 根据索引获取样本 | `dataset[index]` |

下面逐个理解。

## 1\. `__init__()`：初始化数据

先看一个简单的 Python 类：

```python
class Person:

    def __init__(self, name):
        self.name = name

person = Person("Tom")

print(person.name)
```

输出：

```text
Tom
```

这里发生了什么？

第一步：

```python
person = Person("Tom")
```

创建了一个 Person 对象。

第二步：Python 自动执行：

```python
def __init__(self, name):
```

把 `"Tom"` 传给 `name`。

第三步：

```python
self.name = name
```

把传入的名字保存到当前对象中。

所以：

```python
person.name
```

就可以获取 `"Tom"`。

**那么 Dataset 中的 `__init__()` 是做什么的？**

本质上完全一样。

只不过我们保存的不再是一个人的名字，而是训练数据。

例如：

```python
class TextDataset:

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
```

当我们创建对象：

```python
dataset = TextDataset(texts, labels)
```

就会把文本和标签保存到 Dataset 对象中。

注意这里的：

```python
self.texts = texts
self.labels = labels
```

`self` 指向当前创建的对象。

因此，后面无论在哪个方法中，都可以通过：

```python
self.texts
self.labels
```

访问这两份数据。

## 2\. `__len__()`：返回数据集长度

我们希望能够执行：

```python
len(dataset)
```

获得数据集的样本数量。

于是编写：

```python
def __len__(self):
    return len(self.texts)
```

这里有两个不同的 `len`，不要混淆。

外部调用：

```python
len(dataset)
```

Python 就会自动找到 Dataset 类中的：

```python
__len__()
```

而方法内部：

```python
len(self.texts)
```

是计算文本列表本身有多少个元素。

例如：

```python
texts = ["I love AI", "I hate AI", "AI is good"]
```

那么：

```python
len(texts)
```

结果就是：

```text
3
```

因此：

```python
len(dataset)
```

也会返回 `3`。

## 3\. `__getitem__()`：根据索引获取样本

这是今天最重要的方法。

先看一个普通 Python 列表：

```python
numbers = [10, 20, 30]

print(numbers[0])
```

输出：

```text
10
```

这里的 `[0]` 就是通过索引获取第一个元素。

那么：

```python
dataset[0]
```

实际上也采用了类似的设计。

我们编写：

```python
def __getitem__(self, index):

    text = self.texts[index]
    label = self.labels[index]

    return {
        "text": text,
        "label": label
    }
```

逐行解释：

第一行：

```python
def __getitem__(self, index):
```

定义一个特殊方法。

当外部执行：

```python
dataset[2]
```

Python 就会自动调用：

```python
dataset.__getitem__(2)
```

所以这里的 `index` 就是 `2`。

第二行：

```python
text = self.texts[index]
```

根据索引获取对应文本。

第三行：

```python
label = self.labels[index]
```

根据同一个索引获取对应标签。

第四行：

```python
return {
    "text": text,
    "label": label
}
```

把文本和标签组合起来返回。

为什么要组合？

因为模型训练时，需要知道：

**这条输入数据对应的正确答案是什么。**

这两个数据必须保持一一对应。

---

# 四、代码实践：从零编写自己的 Dataset

现在把前面学到的知识组合起来。

今天我们要完成一个真正的 `dataset.py` 文件。

## 1\. 完整代码

```python

import torch
from torch.utils.data import Dataset

# 1. 准备原始数据
texts = [
    "I love AI",
    "This is terrible",
    "AI is amazing",
    "I hate this",
    "Very good work",
    "This is wonderful",
]

labels = [1, 0, 1, 0, 1, 1]

# 2. 自定义 Dataset
class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):

        # 检查文本和标签数量是否一致
        assert len(texts) == len(labels)

        # 保存数据
        self.texts = texts
        self.labels = labels

    def __len__(self):

        # 返回数据集大小
        return len(self.texts)

    def __getitem__(self, index):

        # 根据索引获取文本和标签
        text = self.texts[index]
        label = self.labels[index]

        # 返回一条完整样本
        return {
            "text": text,
            "label": label
        }

# 3. 创建 Dataset 对象
dataset = TextClassificationDataset(texts, labels)

# 4. 测试 Dataset
print("数据集大小：", len(dataset))

print("第一条数据：", dataset[0])

print("第三条数据：", dataset[2])
```

## 2\. 逐段理解代码

**第一部分：导入**

```python
import torch
from torch.utils.data import Dataset
```

这里：

- `torch`：PyTorch 主库。
- `Dataset`：PyTorch 提供的数据集基类。

为什么需要导入 Dataset？

因为我们准备继承它：

```python
class TextClassificationDataset(Dataset):
```

这叫做**类继承**。

意思是：我们定义自己的类，同时继承 PyTorch Dataset 的数据集接口。

**第二部分：数据检查**

```python
assert len(texts) == len(labels)
```

`assert` 是 Python 的断言语句。

它的意思是：

如果后面的条件不成立，就直接报错。

例如：

```python
texts = ["I love AI", "I hate AI"]
labels = [1]
```

这时：

```python
assert len(texts) == len(labels)
```

就会报错。

为什么？

因为有两条文本，却只有一个标签。

如果不检查，后面就可能出现文本与标签错位的问题。

这是一种非常重要的数据质量检查习惯。

**第三部分：保存数据**

```python
self.texts = texts
self.labels = labels
```

把数据保存到当前 Dataset 对象中。

这里要理解一个非常重要的区别：

`texts` 是传入的普通变量。

`self.texts` 是当前对象保存的属性。

之后其他方法都可以通过 `self.texts` 获取数据。

**第四部分：获取长度**

```python
def __len__(self):
    return len(self.texts)
```

告诉 PyTorch，这个数据集有多少条样本。

**第五部分：获取样本**

```python
def __getitem__(self, index):
```

这里的 `index` 是外部传进来的索引。

例如：

```python
dataset[3]
```

就会让 `index = 3`。

接下来：

```python
text = self.texts[index]
label = self.labels[index]
```

获取对应文本和标签。

最后：

```python
return {
    "text": text,
    "label": label
}
```

返回完整样本。

## 3\. 代码运行结果

```text
数据集大小： 6

第一条数据：
{'text': 'I love AI', 'label': 1}

第三条数据：
{'text': 'AI is amazing', 'label': 1}
```

到这里，我们已经成功创建了一个自己的 Dataset。

不过，还有一个非常重要的问题。

## 4\. Dataset 内部究竟是怎么运行的？

我们用一张图把调用关系彻底理清楚。

**创建对象**

`dataset = TextClassificationDataset(texts, labels)`

自动调用 `__init__`\()

`dataset` 对象保存文本和标签

---

**后续外部调用**

**`len(dataset)`**

`__len__()`

返回样本总数

**`dataset[2]`**

`__getitem__(2)`

返回第 3 条样本

这里有个重点：

`__init__()` 只在创建对象时执行一次。

而 `__len__()` 和 `__getitem__()` 可以在后续程序运行中被反复调用。

例如：

```python
dataset[0]
dataset[1]
dataset[2]
dataset[3]
```

每一次索引操作都会触发一次 `__getitem__()`。

这也是为什么 Dataset 能够支持模型训练时反复读取数据。

# 五、训练集与验证集划分

Dataset 还有一个重要的工程知识：数据集划分。

假设我们有 10,000 条训练数据。

不能把所有数据都拿来训练，然后用同一批数据判断模型效果。

因为模型可能只是记住了训练数据，而不是真正学会了规律。

所以通常要划分：

- Training Set：训练集，用于训练模型。
- Validation Set：验证集，用于评估模型表现、调整超参数。
- Test Set：测试集，用于最终评估模型的泛化能力。

例如：

**10,000 条数据的划分示例**

训练集 80%

验证集 10%

测试集 10%

---

| 数据集 | 样本数量 |
|--------|--------|
| Train | 8,000 |
| Validation | 1,000 |
| Test | 1,000 |

PyTorch 提供了一个工具：

```python
from torch.utils.data import random_split
```

可以随机划分 Dataset。

例如：

```python
train_dataset, val_dataset = random_split(
    dataset,
    [4, 2],
    generator=torch.Generator().manual_seed(42)
)
```

逐个理解：

`dataset`：需要划分的原始数据集。

`[4, 2]`：训练集 4 条，验证集 2 条。

`generator`：提供随机数生成器。

`manual_seed(42)`：固定随机种子，让每次运行时划分结果保持一致。

注意：

`random_split()` 返回的是两个 `Subset` 对象，它们通过索引引用原始 Dataset 中的数据，并不是重新复制两份数据。

以后我们可以这样使用：

```python
print(len(train_dataset))  # 4
print(len(val_dataset))    # 2

print(train_dataset[0])
print(val_dataset[0])
```

**工程注意事项：**

1. 训练集和验证集必须互不重叠。
2. 类别不均衡时，不能盲目随机划分，通常需要考虑分层划分（Stratified Split）。
3. 如果数据存在时间顺序，例如金融时间序列，不能随意打乱后划分。
4. 如果预处理过程需要从数据中学习统计信息（例如词表、均值、标准差），应只在训练集上拟合，再应用到验证集和测试集，避免数据泄漏。

今天先掌握 `random_split()`，后续再深入数据划分策略。

---

# 六、独立练习：现在轮到你写代码了

前面是我带着你完成的。接下来我们要检验你是否真的掌握了 Dataset。

(Day 5 实战任务)

## 任务一：自己编写 Dataset

现在给你以下数据：

```python
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
```

要求：

1. 创建 `TextClassificationDataset` 类，继承 `Dataset`。
2. 编写 `__init__()`，保存文本和标签。
3. 编写 `__len__()`，返回样本数量。
4. 编写 `__getitem__()`，根据索引返回文本和标签。
5. 创建 Dataset 对象。
6. 打印数据集长度。
7. 分别获取第 1、3、5 条数据。
8. 使用 `random_split()` 将数据划分为训练集 6 条、验证集 2 条。
9. 打印两个数据集的长度，并分别获取其中一条数据。

## 任务二：理解检查

下面四道题，请你自己判断。

原窗口此处为交互知识检测，题目和作答未提供可读取内容。

# 七、今天必须真正掌握的知识

在进入 Day 6 之前，你需要能够不看资料回答以下问题：

| 核心知识 | 验收标准 |
|--------|--------|
| Dataset 是什么？ | 能解释它负责数据读取与样本获取 |
| `__init__()` | 知道如何初始化、保存数据 |
| `__len__()` | 知道如何返回样本数量 |
| `__getitem__()` | 能独立实现索引取样 |
| `self` | 理解对象属性与方法 |
| 数据集划分 | 会使用 `random_split()` |

**今天最重要的一句话：**

Dataset 的本质就是：**把原始数据封装成一个能够通过索引获取样本、通过 `len()` 获取数据量的对象。**

再提醒你一下，今天我们暂时只返回原始文本和标签，还没有进行 Tokenization，也没有进行 Batch 组织。

后面三天才会逐步解决这些问题。

---

**现在开始实战。**

先不要复制我上面的完整代码。请你自己新建 `dataset.py`，完成任务一，然后把代码发给我。

我会像代码 Review 一样逐行检查：不仅看能不能运行，还会检查代码结构、Python 语法、数据对应关系和工程习惯。通过后再进行 Day 5 验收，不急着进入 Day 6。


---

# 补充讲解 · 继承 Dataset 到底获得什么

你这个问题问到了 **Dataset 的设计本质**。而且你的判断是对的：

**即使不继承 `Dataset`，我们之前写的那个类也可以正常运行！**

所以关键问题不是“为什么必须继承”，而是：

**既然不继承也能运行，那 PyTorch 为什么还提供 Dataset？**

我们通过例子理解。

## 一、先看：不继承 Dataset 可以吗？

完全可以。

```python
class MyDataset:

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "label": self.labels[index]
        }

texts = ["I love AI", "I hate AI"]
labels = [1, 0]

dataset = MyDataset(texts, labels)

print(len(dataset))  # 2
print(dataset[0])    # {'text': 'I love AI', 'label': 1}
```

这个类没有继承 `Dataset`，但是完全能工作。

甚至还能这样：

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2)

for batch in loader:
    print(batch)
```

也能正常加载数据！

**这是因为 PyTorch 的 DataLoader 主要依赖对象提供的接口，而不是强制要求它继承 Dataset。**

## 二、那继承 Dataset 到底有什么用？

我们打个比方。

假设你要给公司招聘一个员工，岗位要求是：

1. 能写代码。
2. 能进行代码审查。
3. 能修复 Bug。

有两种方式：

**方式 A：** 随便找一个人，只要他具备这三种能力，就可以胜任。

**方式 B：** 找一个符合公司统一岗位规范的正式工程师。

两个人可能完成完全相同的工作。

但方式 B 的好处是，其他部门看到他的岗位类型，就知道他遵循什么规范，也更方便接入公司的管理体系。

PyTorch 的 Dataset 也是类似的设计。

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    ...
```

继承后，你的类就成为 PyTorch Dataset 类型体系的一部分。

它的主要价值是：

- **统一接口规范：** 明确这是一个按索引获取样本的数据集。
- **代码可读性：** 其他开发者一看就明白它的用途。
- **框架兼容性：** 可以使用 Dataset 类型相关的检查、扩展和一些框架机制。
- **工程扩展性：** 可以继承、重写 Dataset 提供的功能。

但必须强调：

`Dataset` 并不会自动帮你读取文件、切分文本、生成标签或者完成数据预处理。

这些逻辑依然需要你自己编写。

## 三、从源码看，你就彻底明白了

PyTorch 的 `Dataset` 基类可以简化理解成：

```python
class Dataset:

    def __getitem__(self, index):
        raise NotImplementedError
```

它实际上没有替你实现具体的数据读取功能。

这里的：

```python
raise NotImplementedError
```

意思是：

**这个方法需要由子类实现；如果没有实现却调用它，就会报错。**

因此，当你这样写：

```python
class TextDataset(Dataset):

    def __getitem__(self, index):
        return self.texts[index]
```

你相当于告诉 PyTorch：

“我遵循 Dataset 的规范，具体怎么读取数据，由我自己实现。”

注意，PyTorch 的实际 Dataset 基类还包含其他接口和行为，上面只是为了理解而做的简化。

## 四、两种写法的真正区别

| 对比 | 不继承 Dataset | 继承 Dataset |
|--------|--------|--------|
| 可以保存数据 | ✅ | ✅ |
| 可以实现 `__getitem__` | ✅ | ✅ |
| 可以实现 `__len__` | ✅ | ✅ |
| 可以配合 DataLoader | ✅ 通常可以 | ✅ |
| 属于 PyTorch Dataset 类型 | ❌ | ✅ |
| 具有 Dataset 基类提供的规范和扩展接口 | ❌ | ✅ |
| 工程中是否推荐 | 可以，但较少采用 | **推荐** |

所以你之前感觉 Dataset 没什么用，是因为：

**在我们目前这个简单例子里，继承确实没有带来明显的功能变化。**

这是正常的。

等你学习 DataLoader、IterableDataset、不同数据集的组合以及更复杂的数据加载机制时，就会更容易体会到 PyTorch 为什么需要统一的数据集体系。

---

# 五、Day 5 最终完整代码：dataset.py

这份代码包含今天的全部核心内容：Dataset 定义、数据检查、样本获取、数据集遍历、训练集和验证集划分。

你可以直接保存为 `dataset.py`，独立运行。

```python

import torch
from torch.utils.data import Dataset, random_split

# ======================================
# 1. 准备原始数据
# ======================================

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

# ======================================
# 2. 定义文本分类 Dataset
# ======================================

class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):

        # 检查数据是否匹配
        if len(texts) != len(labels):
            raise ValueError("文本和标签数量必须一致")

        # 保存文本和标签
        self.texts = texts
        self.labels = labels

    def __len__(self):

        # 返回数据集的样本数量
        return len(self.texts)

    def __getitem__(self, index):

        # 取出指定索引的文本
        text = self.texts[index]

        # 取出对应的标签
        label = self.labels[index]

        # 返回一条完整样本
        sample = {
            "text": text,
            "label": label
        }

        return sample

# ======================================
# 3. 创建 Dataset 对象
# ======================================

dataset = TextClassificationDataset(texts, labels)

print("数据集大小:", len(dataset))

# ======================================
# 4. 根据索引获取样本
# ======================================

print("\n第一条样本:")
print(dataset[0])

print("\n第三条样本:")
print(dataset[2])

# ======================================
# 5. 遍历整个 Dataset
# ======================================

print("\n全部样本:")

for i in range(len(dataset)):
    sample = dataset[i]

    print(
        "索引:", i,
        "文本:", sample["text"],
        "标签:", sample["label"]
    )

# ======================================
# 6. 划分训练集和验证集
# ======================================

train_dataset, val_dataset = random_split(
    dataset,
    [6, 2],
    generator=torch.Generator().manual_seed(42)
)

print("\n训练集大小:", len(train_dataset))
print("验证集大小:", len(val_dataset))

# ======================================
# 7. 查看划分后的数据
# ======================================

print("\n训练集:")

for i in range(len(train_dataset)):
    print(train_dataset[i])

print("\n验证集:")

for i in range(len(val_dataset)):
    print(val_dataset[i])
```

## 六、最后用一句话区分三个概念

你接下来会学习的三个机制，可以先这样理解：

**Dataset**

定义如何获取一条样本

`dataset[0]`

**DataLoader**

控制取样、组成 Batch 和加载流程

`batch_size = 4`

**Data Collator**

整理一批样本，例如动态 Padding

`input_ids: [B, T]`

注意：实际上是 DataLoader 在组装 Batch 的过程中调用 `collate_fn`（通常由 Data Collator 实现），而不是先生成完整 Batch 再单独执行 Collator。

**最后调整一下后续学习方式：** 每一天仍然按照“例子引入 → 设计动机 → 专业概念 → 代码实践 → 掌握检查”的顺序来讲，但不再给你布置必须提交的作业。每一天都会提供完整、可独立运行的 `.py` 代码，并重点讲清楚 Python 语法、PyTorch 调用关系及工程细节。


## 追问一 · Dataset 的职责

**Dataset 的作用就是：统一管理数据，让程序知道怎么获取一条样本。**

比如你有 1000 条文本：

```python
dataset[0]  # 获取第1条文本及标签
dataset[1]  # 获取第2条文本及标签
len(dataset) # 获取数据总量
```

**那为什么要继承 `Dataset`？**

其实你说得对：不继承，只要自己实现了这些方法，也能正常工作。

继承 `Dataset` 主要是为了**遵循 PyTorch 统一的数据集规范**，方便与其他工具配合。

所以记住两句话：

- **Dataset：** 管理数据，规定怎么取出一条样本。
- **继承 Dataset：** 表明你的类遵循 PyTorch 的数据集规范，并不是自动获得数据读取能力。

**它本质上是一个统一接口，而不是帮你处理数据的工具。**


## 追问二 · 接口规范体现在哪里

体现在 **`Dataset` 规定了数据集应该提供 `__getitem__()` 接口**，而我们继承后自己实现它：

```python
class MyDataset(Dataset):
    def __getitem__(self, index):
        return self.texts[index]
```

但注意：

1. **继承 Dataset**：表明这个类属于 PyTorch 的 Dataset 类型。
2. **实现 `__getitem__()` 和 `__len__()`**：让它能按索引取数据、返回数据量。

**关键：真正遵循接口规范，靠的是实现这些方法，而不只是写了 `(Dataset)`。** 不继承也可以实现相同接口。


## 追问三 · 谁让方括号调用 __getitem__

因为 **`__getitem__()` 是 Python 自己规定的特殊方法，不是 PyTorch Dataset 规定的。**

Python 天生就知道：

```python
dataset[0]       # 自动调用 __getitem__(0)
len(dataset)     # 自动调用 __len__()
```

**所以不需要继承 Dataset，只要类里定义了这些方法，Python 就能自动调用。**

继承 `Dataset` 只是为了接入 PyTorch 的数据集类型体系，而不是让 Python 认识这些方法。

---

## 与归档练习对应的说明

[code.py](code.py) 来自本地 dataset.py，使用6条原始文本和4/2划分，原讲解后半部分的最终代码使用8条文本和6/2划分。两组数据对应各自的示例。

归档时将 __len__ 中的全局 texts 改为 self.texts，保证不同实例返回自身数据长度，并补充文本/标签数量的 ValueError 检查。原本地文件保留。
