# Day 17 · Gradient Accumulation（梯度累积）

> 本页保留目前可取得的三段完整追问、归档代码和运行说明。会话接口未返回较早的 Day17 主讲解，该部分待补。原始学习窗口为《学习梯度累积》。

## 追问一 · 大 Batch 平均、小 Batch 缩放与累积分别发生在哪里

假设大 Batch 有 8 条样本，小 Batch 有 2 条样本，累积 4 次。

### 1\. 大 Batch 的平均在哪里？

```python
loss = criterion(logits, batch_y)
loss.backward()
```

`criterion` 默认求平均，所以：

$$
Loss=\frac{L_1+L_2+\cdots+L_8}{8}
$$

执行 `backward()`，得到梯度：

$$
w.grad=\frac{g_1+g_2+\cdots+g_8}{8}
$$

### 2\. 小 Batch 的平均和缩放在哪里？

```python
loss = criterion(logits, batch_y)  # 对2条样本求平均
loss = loss / 4                  # 额外缩放
loss.backward()
```

第一行计算：

$$
Loss=\frac{L_1+L_2}{2}
$$

第二行计算：

$$
Loss=\frac{L_1+L_2}{2\times4}=\frac{L_1+L_2}{8}
$$

第三行反向传播：

$$
w.grad=\frac{g_1+g_2}{8}
$$

注意：`backward()` 本身不会主动求平均，而是对已经求平均、缩放后的 Loss 进行求导。

### 3\. 梯度累积在哪里？

**就在 `loss.backward()` 内部。** PyTorch 默认将新计算的梯度累加到 `.grad` 中，而不是覆盖。

| 循环 | 本次计算的梯度 | `w.grad` 中的梯度 |
|--------|--------|--------|
| 第1次 | $\frac{g_1+g_2}{8}$ | $\frac{g_1+g_2}{8}$ |
| 第2次 | $\frac{g_3+g_4}{8}$ | $\frac{g_1+\cdots+g_4}{8}$ |
| 第3次 | $\frac{g_5+g_6}{8}$ | $\frac{g_1+\cdots+g_6}{8}$ |
| 第4次 | $\frac{g_7+g_8}{8}$ | $\frac{g_1+\cdots+g_8}{8}$ |

最后执行 `optimizer.step()`，更新参数。

### 4\. 对应代码

```python
optimizer.zero_grad()

for step, (x, y) in enumerate(loader):

    logits = model(x)

    loss = criterion(logits, y)  # 求平均，除以2

    loss = loss / 4              # 缩放，再除以4

    loss.backward()              # 求导，并自动累积梯度

    if (step + 1) % 4 == 0:
        optimizer.step()         # 更新参数
        optimizer.zero_grad()    # 清空梯度
```

**本质上，大 Batch 是直接除以 8，小 Batch 是先除以 2，再除以 4。最终累积得到的梯度在数学上是一样的。**

## 追问二 · 为什么本次只有两条样本，梯度却除以8

因为在每次 `for` 循环中，实际上执行了两次除法：

```python
loss = criterion(logits, y)  # ① 除以2，因为当前有2条样本
loss = loss / 4              # ② 除以4，因为要累积4次
loss.backward()              # ③ 求导并累积梯度
```

第一次除法：

$$
loss=\frac{L_1+L_2}{2}
$$

第二次除法：

$$
loss=\frac{L_1+L_2}{2\times4}
=\frac{L_1+L_2}{8}
$$

反向传播后，本次产生的梯度自然就是：

$$
\frac{g_1+g_2}{8}
$$

**每次循环确实只处理2条样本。除以8并不是因为当前有8条样本，而是因为我们额外手动除以了4，为最终8条样本的梯度求平均做准备。**

## 追问三 · 为什么还要乘 batch_size

因为 `loss` 是当前 Batch 的损失**平均值，而不是总和**。

假设当前 Batch 有 2 条样本，整个累积周期有 8 条样本。

```python
loss = criterion(logits, batch_y)
```

此时：

$$
loss=\frac{L_1+L_2}{2}
$$

然后：

```python
scaled_loss = loss * batch_size / group_size
```

实际上就是：

$$
\frac{L_1+L_2}{2}\times 2\div 8
=\frac{L_1+L_2}{8}
$$

- `* batch_size`：将平均损失还原为损失总和。
- `/ group_size`：按照整个累积周期的样本数求平均。

如果每个小 Batch 都一样大，那么：

```python
scaled_loss = loss * 2 / 8
```

完全等价于：

```python
scaled_loss = loss / 4
```

**之所以写得复杂一点，就是为了兼容最后一个 Batch 样本数量不足的情况。**

## 归档代码与尾周期

本地 gradient_accumulation.py 使用300条样本、Micro Batch Size=32、累积4次，完整周期128条。前两周期各128条，最后一个周期为32+12=44条，每轮分别在第4、8、10个 Batch 后更新，合计3次。

scaled_loss 按 loss*batch_size/group_size 加权，最后周期分别使用32/44和12/44。统计日志继续使用原始 loss，乘当前样本数后累计，最后除以300。

这份 group_size 计算针对当前固定批大小、drop_last=False 的普通 Dataset，换采样方式或丢弃尾批时需按实际样本数调整。

## 完整归档程序

下方是本地练习的归档版本，修正了 scaled_loss 漏除 group_size 的问题。


```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)  # 注意：这里是函数调用，不是赋值

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

# 三类数据拼接后，x.shape = [300, 2]
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
# 3. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("样本总数:", len(dataset))
print("每个 Epoch 的 Batch 数:", len(loader))

# ============================================
# 4. 定义分类模型
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
# 6. 训练参数
# ============================================

num_epochs = 10

# 【Day17 新增】梯度累积次数
accumulation_steps = 4

# 每次 Forward 处理 32 条数据
micro_batch_size = loader.batch_size

# 一次更新最多对应 128 条数据
effective_batch_size = (
    micro_batch_size * accumulation_steps
)

print("Micro Batch Size:", micro_batch_size)
print("Accumulation Steps:", accumulation_steps)
print("Effective Batch Size:", effective_batch_size)


# ============================================
# 7. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss=0
    total_correct=0
    total_samples=0

    # 【Day17 修改】
    # 一个 Epoch 开始时，清零上次的梯度
    optimizer.zero_grad()

    for batch_index,(batch_x,batch_y) in enumerate(loader):

        # 1. Forward
        logits=model(batch_x)

        # 2. Calculate Loss
        # CrossEntropyLoss 默认返回当前 Batch 的平均 Loss
        loss = criterion(logits, batch_y)

        # 3. 获取当前 Micro Batch 样本数
        batch_size=batch_x.size(0)

        # ====================================
        # 【Day17 新增】计算当前累积周期的样本数
        # ====================================

        # 当前属于第几个累积周期（从 0 开始）
        group_index=batch_index//accumulation_steps

        # 当前累积周期的起始样本位置
        group_start=group_index * effective_batch_size

        # 当前周期实际包含多少条样本
        # 最后一个周期可能不足 128 条
        group_size=min(
            effective_batch_size,
            len(dataset)-group_start
        )

        # ====================================
        # 【Day17 新增】缩放 Loss
        # ====================================

        # 当前 Batch 的平均 Loss
        # 乘以当前 Batch 样本数，再除以整个周期的样本数
        scaled_loss=loss*batch_size/group_size


        # 4. Backward
        # 多次 backward() 将梯度累积到 .grad
        scaled_loss.backward()

        # ====================================
        # 【Day17 新增】判断是否更新参数
        # ====================================

        # 是否达到设定的累积次数
        is_full_cycle = (
            (batch_index + 1) % accumulation_steps == 0
        )

        # 是否已经到达最后一个 Batch
        is_last_batch = (
            (batch_index + 1) == len(loader)
        )

        # 满足任意条件，就更新一次参数
        if is_full_cycle or is_last_batch:

            optimizer.step()

            # 更新完成后，清零累积梯度
            optimizer.zero_grad()

            # 打印梯度累积信息
            if epoch == 0:
                print(
                    f"\n【Day17】完成一次参数更新 "
                    f"当前 Batch: {batch_index + 1} "
                    f"本次累积样本数: {group_size}"
                )

        # ====================================
        # 5. 统计训练指标（保留原来的逻辑）
        # ====================================

        # 注意：这里用原始 loss，不是 scaled_loss
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        correct = (predictions == batch_y).sum().item()

        total_correct += correct

        total_samples += batch_size

        # 第一轮打印部分 Batch 信息
        if epoch == 0 and batch_index < 2:

            print(f"\nBatch {batch_index + 1}")

            print("batch_x shape:", batch_x.shape)
            print("batch_y shape:", batch_y.shape)

            print("logits shape:", logits.shape)
            print("当前 Batch Loss:", loss.item())

    # ========================================
    # 8. 计算 Epoch 指标
    # ========================================

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )

# ============================================
# 9. 训练结束
# ============================================

print("\n训练完成！")
```
