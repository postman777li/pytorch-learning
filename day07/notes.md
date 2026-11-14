# Day 7：Tokenization —— 从文本到模型输入

> 来源为《学习Dataset数据集》的原讲解，保留原文顺序、例子和逐步代码说明。交互组件改为静态表格，公式适配 GitHub Markdown。可运行代码为本目录的 [code.py](code.py)，原始文件名在讲解中保留。

今天正式进入 **PyTorch 数据处理模块 Day 7**。

前两天我们已经掌握：

- **Day 5 Dataset**：如何根据索引获取一条样本。
- **Day 6 DataLoader**：如何把多条样本组成一个 Batch。

但现在还有一个关键问题：

**模型不能直接理解 `"I love AI"` 这样的字符串，那么如何把文本变成神经网络可以处理的数字？**

这就是今天要解决的问题。

**Day 7 · 学习目标** (开始学习)

| 核心知识 | 今天必须掌握 |
|--------|--------|
| Tokenizer | 文本为什么需要分词 |
| Token / Token ID | 文本如何变成数字 |
| Vocabulary | 词表是什么 |
| Padding | 不同长度的文本如何对齐 |
| Truncation | 文本太长怎么办 |
| Attention Mask | 如何区分真实 Token 和填充 |
| 工程实践 | 编写完整 `tokenization.py` |

按照你的学习习惯，今天仍然采用：

**具体例子 → 设计动机 → 专业概念 → 底层原理 → 代码实践。**

最后提供完整、可直接运行的 `tokenization.py`，不留必须提交的作业。

---

## 一、具体例子：为什么需要 Tokenizer？

假设我们有三条文本：

```python
texts = [
    "I love AI",
    "I love deep learning",
    "Hello"
]
```

现在我们要把它们输入 Transformer。

但是，神经网络做的是什么？

矩阵乘法：

$$
Y=XW
$$

这里的 $X$ 必须是数值 Tensor，而不能是字符串。

所以：

```python
"I love AI"
```

不能直接输入 Transformer。

我们需要把它转换成：

```python
[2, 3, 4]
```

这些数字叫做 **Token ID**。

于是整个过程变成：

**原始文本**
`"I love AI"`

**Tokenization**
`["I", "love", "AI"]`

**Token ID 映射**
`[2, 3, 4]`

**Embedding**
`Tensor [3, 512]`

每个 Token ID 被映射为 512 维向量

这里你应该很熟悉最后一步，因为你之前已经学过 Transformer 的 Embedding。

今天我们主要学习前两步：

**文本 → Token → Token ID。**

## 二、设计动机：为什么不直接给每个单词编号？

最简单的想法是建立一个词表。

例如：

```python
vocab = {
    "I": 2,
    "love": 3,
    "AI": 4,
    "deep": 5,
    "learning": 6,
    "Hello": 7
}
```

于是：

```python
"I love AI"
```

变成：

```python
[2, 3, 4]
```

这当然可以。

但假设遇到一个词：

```python
"unbelievable"
```

而词表里没有怎么办？

或者遇到：

```python
"ChatGPT"
```

同样不在词表中。

如果所有单词都必须提前放进词表，会遇到两个问题：

1. 单词数量庞大，词表容易膨胀。
2. 新单词、拼写变化、人名等很难全部覆盖。

因此，现代大模型通常采用 **Subword Tokenization（子词切分）**，而不是简单地按照空格拆单词。

例如，某个 Tokenizer 可能将：

```text
unbelievable
```

切分成类似：

```text
un + believ + able
```

具体怎么切，取决于 Tokenizer 的词表和切分算法。

常见算法包括 BPE、WordPiece、Unigram 等。

你现在只需要先理解：

**Token 不一定等于单词，它也可以是单词的一部分、标点，甚至其他文本片段。**

---

## 三、专业概念：Tokenizer、Token、Token ID、Vocabulary

这四个概念必须区分清楚。

| 概念 | 含义 | 例子 |
|--------|--------|--------|
| Tokenizer | 文本编码、解码的工具 | `tokenizer(text)` |
| Token | 文本切分后的单位 | `"love"` |
| Token ID | Token 对应的整数编号 | `3` |
| Vocabulary | Token 与 ID 的映射词表 | `{"love": 3}` |

举个例子：

```python
tokens = ["I", "love", "AI"]
```

词表：

```python
vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "I": 2,
    "love": 3,
    "AI": 4
}
```

逐个查询：

```python
vocab["I"]       # 2
vocab["love"]    # 3
vocab["AI"]      # 4
```

得到：

```python
input_ids = [2, 3, 4]
```

注意：

**Token ID 只是索引，不代表语义。**

例如 `AI=4` 不代表 AI 的含义是数字 4。

真正将 Token 转换成向量表示的是后面的 Embedding 层：

```python
embedding = torch.nn.Embedding(
    num_embeddings=5,
    embedding_dim=512
)
```

其中：

- `num_embeddings=5`：词表共有 5 个 Token。
- `embedding_dim=512`：每个 Token 对应一个 512 维向量。

---

## 四、Padding：三条文本长度不同怎么办？

现在我们已经能把文本转换成 Token ID。

假设编码结果如下：

```python
[
    [2, 3, 4],
    [2, 3, 5, 6],
    [7]
]
```

问题来了。

第一条长度是 3。

第二条长度是 4。

第三条长度是 1。

但我们希望把三条文本组成一个形状规则的 Tensor：

```python
[B, T]
```

其中：

- `B`：Batch Size。
- `T`：序列长度。

不同长度的列表不能直接组成普通的二维数值 Tensor。

所以需要 **Padding（填充）**。

### 1\. Padding 的原理

我们使用 `[PAD] = 0`。

将所有序列补齐到同样长度，比如 4：

| 样本 | Padding 前 | Padding 后 |
|---|---|---|
| 1 | `[2,3,4]` | `[2,3,4,0]` |
| 2 | `[2,3,5,6]` | `[2,3,5,6]` |
| 3 | `[7]` | `[7,0,0,0]` |

表中0为补充的 PAD。

最终：

```python
input_ids = [
    [2, 3, 4, 0],
    [2, 3, 5, 6],
    [7, 0, 0, 0]
]
```

就可以转换为：

```python
torch.tensor(input_ids)
```

Shape：

```text
torch.Size([3, 4])
```

也就是：

- 3 条样本。
- 每条序列长度统一为 4。

### 2. 统一长度与 Tensor Shape

| 统一长度 | Batch Shape | 处理方式 |
|---|---|---|
| 3 | `[3,3]` | 长度4的样本截断，较短样本补齐 |
| 5 | `[3,5]` | 三条样本均补齐到5 |
| 7 | `[3,7]` | 三条样本均补齐到7 |

下节介绍超长序列的截断。

---

## 五、Truncation：文本太长怎么办？

Padding 是文本太短时补零。

那么文本太长怎么办？

这就需要 **Truncation（截断）**。

假设模型最多接受 5 个 Token：

```python
max_length = 5
```

但有一条文本编码后是：

```python
[2, 3, 4, 5, 6, 7, 8]
```

长度是 7，超过限制。

我们可以截断为：

```python
[2, 3, 4, 5, 6]
```

最简单的 Python 写法：

```python
ids = ids[:max_length]
```

这句的意思是：

从列表开头取出最多 `max_length` 个元素。

**为什么需要 Truncation？**

因为 Transformer 存在最大上下文长度限制，而且序列越长，通常意味着更大的计算和显存开销。

但截断也可能丢失重要信息，因此实际项目里不能随意设置特别短的 `max_length`。

---

## 六、Attention Mask：Padding 之后，模型怎么知道哪些是填充？

这是今天非常重要的一个概念。

假设：

```python
input_ids = [2, 3, 4, 0, 0]
```

这里：

- `2,3,4`：真实 Token。
- `0,0`：Padding。

如果不额外告诉模型哪些位置是填充，模型就可能把 Padding 位置也纳入计算。

所以我们需要构造：

```python
attention_mask = [1, 1, 1, 0, 0]
```

其中：

- `1`：有效 Token。
- `0`：Padding Token。

例如：

| input\_ids | 2 | 3 | 4 | 0 | 0 |
|--------|--------|--------|--------|--------|--------|
| attention\_mask | 1 | 1 | 1 | 0 | 0 |

这就是 **Attention Mask（注意力掩码）**。

这里的 Mask 是数据有效性标记，不要和你之前学习的 GPT Causal Mask 混淆。

- **Padding Mask**：标记哪些位置是填充。
- **Causal Mask**：阻止模型看到未来 Token。

实际 Transformer 可以结合两种限制来计算 Attention。

另外，`attention_mask` 不会自动让所有模型忽略 Padding，模型必须正确使用它。

---

# 七、代码实践：自己从零编写一个简单 Tokenizer

为了理解底层原理，我们先不用现成的 Hugging Face Tokenizer。

而是自己写一个最简单的版本：

**文本按空格拆分 → 查询词表 → 转换 Token ID → 截断 → Padding → Attention Mask。**

注意：这是教学版的空格分词 Tokenizer，不是工业级 BPE Tokenizer。它用来帮助你理解数据管线最合适。

## 第一步：构建词表

```python
vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "I": 2,
    "love": 3,
    "AI": 4,
    "deep": 5,
    "learning": 6,
    "Hello": 7
}
```

其中两个特殊 Token：

- `[PAD]`：用于 Padding。
- `[UNK]`：表示词表中不存在的 Token。

例如：

```python
vocab.get("love", 1)
```

输出：

```text
3
```

而：

```python
vocab.get("Python", 1)
```

输出：

```text
1
```

因为 `"Python"` 不在词表中，所以返回默认值 `1`。

这里的 `dict.get(key, default)` 是 Python 字典的方法。

## 第二步：文本切分

```python
text = "I love AI"

tokens = text.split()

print(tokens)
```

输出：

```python
["I", "love", "AI"]
```

`split()` 默认按空白字符拆分字符串。

## 第三步：转换为 Token ID

```python
ids = []

for token in tokens:
    token_id = vocab.get(token, 1)
    ids.append(token_id)

print(ids)
```

输出：

```python
[2, 3, 4]
```

这里 `append()` 就是在列表末尾添加一个元素。

## 第四步：截断

```python
max_length = 5

ids = ids[:max_length]
```

如果长度超过 5，就只保留前 5 个 Token。

## 第五步：生成 Attention Mask

```python
attention_mask = [1] * len(ids)
```

假设：

```python
ids = [2, 3, 4]
```

那么：

```python
[1] * 3
```

就是：

```python
[1, 1, 1]
```

## 第六步：Padding

```python
padding_length = max_length - len(ids)

ids = ids + [0] * padding_length

attention_mask = attention_mask + [0] * padding_length
```

假设当前长度是 3，最大长度是 5：

```python
padding_length = 5 - 3
```

结果为 2。

于是：

```python
ids = [2, 3, 4, 0, 0]
```

同时：

```python
attention_mask = [1, 1, 1, 0, 0]
```

现在我们就实现了最核心的 Tokenizer 编码逻辑。

---

# 八、Day 7 完整代码：tokenization.py

下面这份代码可以直接保存为 `tokenization.py` 运行。

它包含：

1. 原始文本。
2. 词表构建。
3. Tokenization。
4. Token ID 转换。
5. Truncation。
6. Padding。
7. Attention Mask。
8. 将多条文本转换成 PyTorch Tensor。
9. 查看最终输入 Shape。

```python

import torch

# ==========================================
# 1. 准备原始文本
# ==========================================

texts = [
    "I love AI",
    "I love deep learning",
    "Hello"
]

# ==========================================
# 2. 构建词表 Vocabulary
# ==========================================

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "I": 2,
    "love": 3,
    "AI": 4,
    "deep": 5,
    "learning": 6,
    "Hello": 7
}

PAD_ID = vocab["[PAD]"]
UNK_ID = vocab["[UNK]"]

# ==========================================
# 3. 定义简单 Tokenizer
# ==========================================

class SimpleTokenizer:

    def __init__(self, vocab):

        self.vocab = vocab

        self.pad_id = vocab["[PAD]"]
        self.unk_id = vocab["[UNK]"]

    def tokenize(self, text):

        # 按空白字符切分文本
        tokens = text.split()

        return tokens

    def convert_tokens_to_ids(self, tokens):

        ids = []

        for token in tokens:

            # 查找 token 对应的 ID
            token_id = self.vocab.get(
                token,
                self.unk_id
            )

            ids.append(token_id)

        return ids

    def encode(self, text, max_length=5):

        # Step 1: 文本切分
        tokens = self.tokenize(text)

        # Step 2: 转换为 Token ID
        input_ids = self.convert_tokens_to_ids(tokens)

        # Step 3: Truncation
        input_ids = input_ids[:max_length]

        # Step 4: 构造 Attention Mask
        attention_mask = [1] * len(input_ids)

        # Step 5: 计算 Padding 长度
        padding_length = max_length - len(input_ids)

        # Step 6: Padding
        input_ids = (
            input_ids
            + [self.pad_id] * padding_length
        )

        attention_mask = (
            attention_mask
            + [0] * padding_length
        )

        # Step 7: 返回编码结果
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# ==========================================
# 4. 创建 Tokenizer
# ==========================================

tokenizer = SimpleTokenizer(vocab)

# ==========================================
# 5. 单条文本编码
# ==========================================

text = "I love AI"

tokens = tokenizer.tokenize(text)

print("原始文本:", text)
print("Tokens:", tokens)

encoded = tokenizer.encode(
    text,
    max_length=5
)

print("Input IDs:", encoded["input_ids"])
print("Attention Mask:", encoded["attention_mask"])

# ==========================================
# 6. 批量编码文本
# ==========================================

all_input_ids = []
all_attention_masks = []

for text in texts:

    encoded = tokenizer.encode(
        text,
        max_length=5
    )

    all_input_ids.append(
        encoded["input_ids"]
    )

    all_attention_masks.append(
        encoded["attention_mask"]
    )

# ==========================================
# 7. 转换为 PyTorch Tensor
# ==========================================

input_ids_tensor = torch.tensor(
    all_input_ids,
    dtype=torch.long
)

attention_mask_tensor = torch.tensor(
    all_attention_masks,
    dtype=torch.long
)

# ==========================================
# 8. 查看最终结果
# ==========================================

print("\n===== Batch 输入 =====")

print("input_ids:")
print(input_ids_tensor)

print("\nattention_mask:")
print(attention_mask_tensor)

print("\ninput_ids shape:")
print(input_ids_tensor.shape)

print("\nattention_mask shape:")
print(attention_mask_tensor.shape)

# ==========================================
# 9. 模拟 Embedding 层
# ==========================================

embedding = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=8,
    padding_idx=PAD_ID
)

embedded = embedding(input_ids_tensor)

print("\nEmbedding 输出 Shape:")
print(embedded.shape)
```

## 九、这份代码实际输出什么？

最重要的是最终 Batch。

`input_ids`：

```text
tensor([
    [2, 3, 4, 0, 0],
    [2, 3, 5, 6, 0],
    [7, 0, 0, 0, 0]
])
```

`attention_mask`：

```text
tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0]
])
```

Shape：

```text
input_ids:       torch.Size([3, 5])
attention_mask:  torch.Size([3, 5])
```

再经过：

```python
embedding = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=8
)
```

得到：

```text
torch.Size([3, 5, 8])
```

这个 Shape 应该非常熟悉：

$$
[B,T,C]=[3,5,8]
$$

其中：

- B = 3：三条文本。
- T = 5：每条文本最多五个 Token。
- C = 8：每个 Token 的 Embedding 维度。

这就和你之前学习的 Transformer 输入结构连接起来了。

---

# 十、工业界实际使用什么 Tokenizer？

实际训练大模型时，我们通常不会自己维护一个这样的空格分词词表。

而是使用经过训练的 Tokenizer，例如 Hugging Face Transformers 提供的 `AutoTokenizer`。

先安装：

```bash
pip install transformers
```

然后可以这样使用：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

texts = [
    "I love AI",
    "I love deep learning",
    "Hello"
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=8,
    return_tensors="pt"
)

print(batch["input_ids"])
print(batch["attention_mask"])
print(batch["input_ids"].shape)
```

`AutoTokenizer.from_pretrained(...)` 会加载模型对应的 Tokenizer 配置与词表；首次使用通常需要下载相关文件。

这里有几个关键参数：

| 参数 | 含义 |
|--------|--------|
| `padding=True` | 将当前批次补齐到其中最长的序列 |
| `truncation=True` | 超过最大长度则截断 |
| `max_length=8` | 指定截断长度上限 |
| `return_tensors="pt"` | 返回 PyTorch Tensor |

特别注意：

`padding=True` 不等于一定补齐到 `max_length=8`。

它默认补齐到**本次 Batch 中最长序列的长度**。

如果希望一定补齐到长度 8，可以使用：

```python
padding="max_length"
```

另外，BERT 的真实 Tokenizer 可能添加 `[CLS]`、`[SEP]` 等特殊 Token，所以它的 Token ID 和我们手写教学版的结果不会相同。

---

# 十一、Day 7 最终总结

今天真正需要掌握的是整个数据变化过程：

**原始文本**
`"I love AI"`

**Tokenization**
`["I", "love", "AI"]`

**Token ID**
`[2, 3, 4]`

**Truncation + Padding**
`[2, 3, 4, 0, 0]`

**Attention Mask**
`[1, 1, 1, 0, 0]`

**PyTorch Tensor → Embedding**
`[B, T] → [B, T, C]`

你可以用这张表检查是否掌握：

| 知识点 | 必须理解的核心 |
|--------|--------|
| Tokenizer | 将文本编码成模型使用的 Token ID |
| Vocabulary | Token 与 ID 的映射表 |
| `[UNK]` | 处理词表中未收录的 Token |
| `[PAD]` | 填充较短的序列 |
| Truncation | 截断超过长度限制的序列 |
| Attention Mask | 标记真实 Token 和 Padding |
| `torch.long` | Token ID 通常使用的整数数据类型 |

**Day 7 和 Day 8 的关键区别：**

今天我们是在 `encode()` 里对每条文本使用固定的 `max_length=5`。

但在实际训练时，固定长度 Padding 可能浪费计算资源。

例如，某个 Batch 最长只有 20 个 Token，我们却把所有文本补齐到 512。

因此，**Day 8 Data Collator** 会重点学习：如何等 Dataset 取出多条样本之后，再根据当前 Batch 中最长的序列进行动态 Padding。

届时我们会把 Day 5–8 的代码正式连接起来，形成完整的数据输入管线。

---

## 补充问答 · AutoTokenizer、WordPiece 与词表

对，理解方向是对的，但有一个细节需要区分。

### 1\. AutoTokenizer 是什么？

```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

`AutoTokenizer` 是 Hugging Face Transformers 提供的一个**自动加载 Tokenizer 的工具**，它本身不是分词算法。

`"bert-base-uncased"` 指定了要加载的预训练模型对应的 Tokenizer。

这个 Tokenizer 使用的是 **WordPiece 分词算法**。

### 2\. 它有自己的词表吗？

有！而且每个预训练模型通常都有与之配套的 Tokenizer 和词表。

例如 BERT 有自己的 Vocabulary，里面保存了：

| Token | Token ID（示意） |
|--------|--------|
| `[PAD]` | 0 |
| `[UNK]` | 100 |
| `hello` | 对应编号 |
| `world` | 对应编号 |

实际编号以加载的词表为准。

当你执行：

```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

它会自动加载 BERT 对应的：

- 分词规则（WordPiece）
- 词表（Vocabulary）
- 特殊 Token 配置（如 `[PAD]`、`[CLS]`、`[SEP]`）

另外，`uncased` 表示不区分大小写，通常会先将英文转换为小写。

### 3\. 记住这三个概念的区别

| 名称 | 是什么 |
|--------|--------|
| WordPiece | 分词算法 |
| Vocabulary | Token 与 Token ID 的映射表 |
| AutoTokenizer | 自动加载对应 Tokenizer 的工具 |

**最重要的一点：**

不同模型通常有自己配套的 Tokenizer 和词表，不能随意混用。

例如 BERT 和 GPT 的 Tokenizer 就可能完全不同。同一段文本经过它们处理，得到的 Token 切分和 Token ID 都可能不同。
