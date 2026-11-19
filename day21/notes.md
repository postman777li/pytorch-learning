好，我们正式进入 **模块六：训练工程与综合实战 · Day 21：Checkpoint（模型检查点）**。

这一天非常重要。前面 Day 1–20，你已经学习了模型构建、数据处理、训练循环、优化器、学习率调度、AMP 等知识。

但现在我们要开始从「会训练模型」向「会管理一个真正的模型训练项目」过渡。

今天不追求知识点数量，而是要真正掌握：**如何保存一个正在训练的模型，以及如何让它在程序中断后，从原来的位置继续训练。**

# Day 21：Checkpoint 模型检查点

(今日目标) 能够独立编写模型保存、加载和训练状态恢复代码。

## 第一步：通过具体例子理解问题

假设你正在训练一个大模型。

训练过程是这样的：

| 时间 | 训练状态 |
|--------|--------|
| 09:00 | 开始训练，Epoch 1 |
| 10:00 | 完成 Epoch 1 |
| 11:00 | 完成 Epoch 2 |
| 12:00 | 完成 Epoch 3 |
| 12:30 | 电脑突然断电 |
| 重新启动 | 怎么办？ |

如果你没有保存任何东西，那么之前训练的结果可能全部丢失。

你只能重新从 Epoch 1 开始训练。

这显然很浪费时间。

于是你想到一个办法：

**每训练完一个 Epoch，就把当前模型的状态保存下来。**

例如：

```text
Epoch 1 完成 → 保存 checkpoint
Epoch 2 完成 → 保存 checkpoint
Epoch 3 完成 → 保存 checkpoint
```

假设 Epoch 3 训练完成后电脑断电。

重新启动程序时：

```text
加载 Epoch 3 的 checkpoint
         ↓
恢复模型状态
         ↓
恢复优化器状态
         ↓
从 Epoch 4 继续训练
```

这就是 Checkpoint 要解决的问题。

---

## 第二步：理解设计动机

这里有一个非常重要的问题：

**我们究竟需要保存什么？**

很多初学者会觉得：

> 把模型的参数保存下来不就行了吗？

其实不够。

我们来举个例子。

假设你正在使用 AdamW 优化器训练模型。

经过 3 个 Epoch 后：

- 模型已经学习到了一些东西，参数发生了变化。
- AdamW 已经积累了一些历史梯度信息。
- Learning Rate Scheduler 已经调整过学习率。
- AMP 的 GradScaler 可能已经动态调整过缩放比例。

现在突然中断。

如果你只保存模型参数，那么：

- 模型参数可以恢复。
- AdamW 的历史梯度信息丢失。
- 学习率调度进度丢失。
- AMP 的动态缩放状态丢失。

这意味着：

**模型虽然恢复了，但训练过程并没有完整恢复。**

所以，我们需要保存的不只是模型，而是整个训练过程中的关键状态。

## 第三步：专业概念与原理

### 1\. 什么是 Checkpoint？

Checkpoint，中文通常叫作检查点。

它本质上就是：

**在训练过程中的某个时刻，把模型和训练过程的重要状态打包保存起来。**

我们今天重点掌握以下六项：

**Checkpoint 包含什么？** (6 个核心部分)

---

**1\. Model State**

模型参数及持久化缓冲区

(必需)

**2\. Optimizer State**

优化器的历史状态，例如 AdamW 的动量

(必需)

**3\. Scheduler State**

学习率调度器的当前进度和状态

(必需\*)

**4\. Scaler State**

AMP 动态 Loss Scaling 的状态

(使用 AMP 时)

**5\. Epoch / Global Step**

训练到了第几个 Epoch、执行了多少次更新

(必需)

**6\. Config**

模型配置、学习率、Batch Size 等实验参数

(推荐)

\* 如果不使用 Scheduler，可以不保存 Scheduler State；其他组件同理。

下面我们逐个理解。

### 2\. Model State：模型状态

首先认识一个非常重要的概念：

`state_dict()`

这是 PyTorch 中用于获取模型状态字典的方法。

例如：

```python
model.state_dict()
```

它会返回一个字典，里面保存着模型的参数以及持久化缓冲区。

假设模型是：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 3)
```

执行：

```python
state = model.state_dict()
```

你可以把它理解成：

```python
{
    "fc1.weight": ...,
    "fc1.bias": ...,
    "fc2.weight": ...,
    "fc2.bias": ...
}
```

注意：

这里保存的是参数张量，而不是模型的完整 Python 类定义。

因此，加载时需要先重新创建模型：

```python
model = MyModel()

model.load_state_dict(state)
```

这就把保存的参数重新加载进模型了。

**记住两个方法：**

| 方法 | 作用 |
|--------|--------|
| `model.state_dict()` | 获取模型状态 |
| `model.load_state_dict(state)` | 将状态加载进模型 |

### 3\. Optimizer State：优化器状态

这部分非常重要，尤其是你之前学习过 AdamW。

回忆一下：

AdamW 不仅仅保存模型参数，它还会维护历史梯度的一阶矩、二阶矩等信息。

例如：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)
```

训练过程中，AdamW 会不断更新自己的内部状态。

可以理解为：

```text
模型参数：
    weight、bias

AdamW 内部状态：
    exp_avg
    exp_avg_sq
    step
```

其中：

- `exp_avg`：梯度的一阶动量估计。
- `exp_avg_sq`：梯度的二阶动量估计。
- `step`：参数更新的步数。

如果只恢复模型参数，却重新创建一个 AdamW：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)
```

那么 AdamW 的历史信息就会重新开始积累。

这就不是真正意义上的完整断点续训。

所以需要：

```python
optimizer.state_dict()
```

保存优化器状态。

加载时：

```python
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)
```

**记住：模型参数和优化器状态是两回事，必须分别保存。**

### 4\. Scheduler State：学习率调度器状态

你前面学过：

- StepLR
- CosineAnnealingLR
- Warmup
- Cosine Decay

这些调度器都需要记录自己的运行进度。

例如：

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.1
)
```

假设初始学习率：

$$
lr=0.001
$$

每经过 2 个 Epoch，学习率乘以 0.1。

| Epoch | 学习率 |
|--------|--------|
| 1 | 0.001 |
| 2 | 0.0001 |
| 3 | 0.0001 |
| 4 | 0.00001 |

如果训练到 Epoch 3 时中断，但你没有保存 Scheduler State，那么恢复时调度器可能又从初始进度开始。

所以：

保存：

```python
scheduler.state_dict()
```

加载：

```python
scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)
```

这样才能恢复调度器的进度。

### 5\. Scaler State：混合精度状态

这部分对应你之前学习的 AMP。

回忆一下：

```python
scaler = torch.amp.GradScaler("cuda")
```

GradScaler 主要用于动态调整 Loss Scaling，降低混合精度训练时梯度下溢的风险。

它内部会记录：

- 当前的缩放因子 scale。
- 连续成功更新的次数。
- 是否出现 Inf / NaN 等信息。

如果中断后重新创建 Scaler：

```python
scaler = torch.amp.GradScaler("cuda")
```

那么之前动态调整的状态就丢失了。

因此需要：

```python
scaler.state_dict()
```

保存。

加载：

```python
scaler.load_state_dict(
    checkpoint["scaler_state_dict"]
)
```

如果没有使用 AMP，也可以保存一个禁用状态的 Scaler，其状态字典通常为空。

### 6\. Epoch 和 Global Step

这两个概念需要区分。

| 名称 | 含义 | 举例 |
|--------|--------|--------|
| Epoch | 整个训练集被完整训练了多少轮 | Epoch 3 |
| Global Step | 优化器实际更新了多少次 | Step 1500 |

假设：

- 训练集有 3000 条数据。
- Batch Size = 100。
- 每个 Epoch 有 30 个 Batch。

那么：

训练 1 个 Epoch：

$$
30\text{ 次更新}
$$

训练 3 个 Epoch：

$$
30\times3=90\text{ 次更新}
$$

因此：

```python
checkpoint = {
    "epoch": 3,
    "global_step": 90
}
```

注意：这里的 Global Step 统计的是参数更新次数，不一定等于 Batch 数量。例如使用梯度累积时，两者可能不同。

### 7\. 如何保存 Checkpoint？

现在把上面的内容整合起来。

首先认识两个函数：

```python
torch.save()
torch.load()
```

它们分别负责保存和加载。

例如：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "epoch": epoch,
    "global_step": global_step
}

torch.save(checkpoint, "checkpoint.pth")
```

这就把整个字典保存到了文件中。

加载：

```python
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)
```

这里的 `map_location=device` 很重要。

它的作用是：将读取出来的张量映射到指定设备。

例如：

- 原来在 GPU 上训练。
- 现在想在 CPU 上加载。

就可以指定：

```python
device = torch.device("cpu")
```

避免加载时出现设备不匹配的问题。

---

## 第四步：代码实践

前面理解了原理，现在开始真正写代码。

我们先做三个小练习。

### 练习一：只保存和加载模型参数

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)

# 1. 创建模型
model = MyModel()

# 2. 保存模型参数
torch.save(
    model.state_dict(),
    "model.pth"
)

# 3. 重新创建一个模型
new_model = MyModel()

# 4. 加载参数
state = torch.load("model.pth", weights_only=True)

new_model.load_state_dict(state)

print("模型参数加载成功")
```

这里重点理解：

```python
new_model = MyModel()
```

为什么要重新创建？

因为 `state_dict()` 保存的是参数，不包含完整的模型结构。

所以必须先有一个结构相同的模型，再把参数装进去。

### 练习二：保存完整训练状态

现在把优化器、调度器和 Scaler 一起保存。

```python
checkpoint = {
    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "scheduler_state_dict": scheduler.state_dict(),

    "scaler_state_dict": scaler.state_dict(),

    "epoch": epoch,

    "global_step": global_step
}

torch.save(checkpoint, "checkpoint.pth")
```

注意这里：

`checkpoint` 本身只是一个普通 Python 字典。

我们把不同组件的状态全部放进字典，然后一次性保存。

### 练习三：恢复训练状态

加载时：

```python
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)

scaler.load_state_dict(
    checkpoint["scaler_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1

global_step = checkpoint["global_step"]
```

这里有一个容易出错的地方：

```python
start_epoch = checkpoint["epoch"] + 1
```

为什么加 1？

因为我们约定保存的是已经完成的 Epoch。

假设保存时：

```python
checkpoint["epoch"] = 3
```

说明 Epoch 3 已经完成。

那么恢复后应该：

```python
start_epoch = 4
```

而不是重新训练 Epoch 3。

还有一个工程细节：恢复时必须先重新创建模型、优化器、Scheduler 和 Scaler，再加载它们的状态。尤其是 Scheduler，要先初始化，再加载优化器状态，最后加载 Scheduler 状态，避免 Scheduler 初始化覆盖恢复的学习率。

---

# 第五步：Day 21 完整代码

(今日核心实战) `checkpoint.py`

现在我们做一个真正完整的例子。

目标：

1. 创建一个分类模型。
2. 训练 Epoch 1。
3. 保存 Checkpoint。
4. 模拟程序中断。
5. 重新创建模型和训练组件。
6. 恢复全部训练状态。
7. 从 Epoch 2 继续训练。

这个例子会把今天学到的知识全部串起来。

## 1\. 完整代码

下面代码可以直接复制运行。

```python

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. 基础配置
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

BATCH_SIZE = 32
TOTAL_EPOCHS = 2
CHECKPOINT_PATH = "checkpoint.pth"

# ==========================================
# 2. 准备数据
# ==========================================

torch.manual_seed(42)

X = torch.randn(512, 20)

true_weight = torch.randn(20, 3)

y = (X @ true_weight).argmax(dim=1)

dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# 3. 定义模型
# ==========================================

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. 创建模型及训练组件
# ==========================================

def build_training_objects():

    model = MyModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    return model, optimizer, scheduler, scaler

# ==========================================
# 5. 训练一个 Epoch
# ==========================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            logits = model(inputs)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)

# ==========================================
# 6. 验证模型
# ==========================================

def evaluate(model, loader):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    correct = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)

            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)

    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy

# ==========================================
# 7. 保存 Checkpoint
# ==========================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss
):

    checkpoint = {

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "epoch": epoch,

        "global_step": global_step,

        "best_val_loss": best_val_loss,

        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": 0.001,
            "total_epochs": TOTAL_EPOCHS
        }
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("Checkpoint 保存成功！")

# ==========================================
# 8. 加载 Checkpoint
# ==========================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler
):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    global_step = checkpoint["global_step"]

    best_val_loss = checkpoint["best_val_loss"]

    print("Checkpoint 加载成功！")

    print("恢复 Epoch:", start_epoch)

    print("恢复 Global Step:", global_step)

    return start_epoch, global_step, best_val_loss

# ==========================================
# 9. 第一次训练：训练 Epoch 1
# ==========================================

model, optimizer, scheduler, scaler = build_training_objects()

global_step = 0

best_val_loss = float("inf")

for epoch in range(1, 2):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler
    )

    global_step += len(train_loader)

    val_loss, accuracy = evaluate(
        model,
        val_loader
    )

    scheduler.step()

    best_val_loss = min(best_val_loss, val_loss)

    print(
        f"Epoch: {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step,
        best_val_loss
    )

# ==========================================
# 10. 模拟程序中断
# ==========================================

print("\n========== 模拟程序重新启动 ==========\n")

# 重新创建所有训练组件
model, optimizer, scheduler, scaler = build_training_objects()

# ==========================================
# 11. 恢复训练状态
# ==========================================

start_epoch, global_step, best_val_loss = load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler
)

# ==========================================
# 12. 从 Epoch 2 继续训练
# ==========================================

for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler
    )

    global_step += len(train_loader)

    val_loss, accuracy = evaluate(
        model,
        val_loader
    )

    scheduler.step()

    best_val_loss = min(best_val_loss, val_loss)

    print(
        f"Epoch: {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step,
        best_val_loss
    )

print("\n训练全部完成！")
```

## 2\. 重点代码逐段理解

这份代码比较长，我们重点攻克几个工程中经常使用的地方。

**① 为什么要把创建训练组件单独封装成函数？**

```python
def build_training_objects():

    model = MyModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    return model, optimizer, scheduler, scaler
```

因为训练开始时要创建一次。

程序中断后，也需要重新创建一次。

如果不封装，就需要重复写一遍模型、优化器、调度器、Scaler 的创建代码。

封装之后：

```python
model, optimizer, scheduler, scaler = build_training_objects()
```

一行就能完成。

这是工程代码中非常常见的封装思路。

**② 为什么保存时要传入这么多参数？**

```python
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss
):
```

因为 Checkpoint 并不是只保存模型。

它需要知道整个训练过程的状态。

因此，我们把所有需要保存的组件传进来。

然后：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
}
```

这里的：

```python
"model_state_dict"
```

是字典的 Key。

```python
model.state_dict()
```

是对应的 Value。

这和你之前学习的 Python 字典完全一样。

**③ 为什么重新创建模型之后，还要加载参数？**

看这段：

```python
model, optimizer, scheduler, scaler = build_training_objects()
```

此时创建的是一个全新的模型。

模型参数是随机初始化的。

然后：

```python
model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

这一步才将之前训练过的参数恢复进去。

因此：

- 创建模型：恢复模型结构。
- 加载 state\_dict：恢复模型参数。

两者缺一不可。

**④ 为什么先加载 Optimizer，再加载 Scheduler？**

```python
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)
```

注意：创建 Scheduler 时，它依赖 Optimizer。

恢复时，我们先初始化好 Scheduler，再恢复 Optimizer 状态，最后恢复 Scheduler 状态。

这样可以避免 Scheduler 初始化时对学习率的设置覆盖已经加载的优化器状态。

**⑤ 为什么 Global Step 这样计算？**

```python
global_step += len(train_loader)
```

因为：

```python
len(train_loader)
```

返回一个 Epoch 中有多少个 Batch。

当前代码没有使用梯度累积，每个 Batch 都更新一次参数，所以每个 Epoch 的更新次数就是 Batch 数量。

注意：在真实工程中，如果使用梯度累积，应当按照实际 `optimizer.step()` 的次数更新 Global Step。

---

## 3\. 你运行之后应该看到什么？

程序会先完成 Epoch 1，保存 Checkpoint。

随后模拟程序重新启动。

你应该看到类似：

```text
Epoch: 1 | Train Loss: ... | Val Loss: ... | Accuracy: ...
Checkpoint 保存成功！

========== 模拟程序重新启动 ==========

Checkpoint 加载成功！
恢复 Epoch: 2
恢复 Global Step: 16

Epoch: 2 | Train Loss: ... | Val Loss: ... | Accuracy: ...
Checkpoint 保存成功！

训练全部完成！
```

其中 Global Step 的具体数值由 Batch 数量决定。

你可以特别观察：

1. Epoch 1 保存后，程序重新创建了模型。
2. 恢复时 Epoch 自动变成 2。
3. Optimizer 和 Scheduler 的状态也被恢复。
4. Epoch 2 不需要重新从头开始。

这就是今天最重要的成果。

---

# Day 21 知识总结

今天必须牢固掌握以下内容：

| 知识点 | 必须掌握的内容 |
|--------|--------|
| Checkpoint | 保存训练过程的完整状态 |
| `state_dict()` | 获取组件内部状态 |
| `load_state_dict()` | 恢复组件内部状态 |
| `torch.save()` | 保存 Checkpoint |
| `torch.load()` | 加载 Checkpoint |
| Model State | 恢复模型参数 |
| Optimizer State | 恢复优化器历史信息 |
| Scheduler State | 恢复学习率调度进度 |
| Scaler State | 恢复 AMP 动态缩放状态 |
| Epoch / Global Step | 恢复训练进度 |

还有两个容易混淆的概念：

**模型权重保存**

```python
torch.save(model.state_dict(), "model.pth")
```

主要用于保存模型参数，方便以后加载模型进行推理或微调。

---

**完整训练 Checkpoint**

```python
torch.save(checkpoint, "checkpoint.pth")
```

保存模型、优化器、调度器、Scaler、训练进度等，用于恢复训练。

注意：今天我们还没有实现完全无缝的断点续训。比如随机数状态、DataLoader Shuffle 状态、训练中途某个 Batch 的精确恢复，这些属于 Day 22 的内容。

---

# 今日掌握度检测

在结束 Day 21 前，我希望你能够不看上面的内容回答下面 5 道题。

> 原窗口自测组件没有返回题目，仅保留下方可取得的完成标准。

**今天的完成标准：**

- [ ] 能解释为什么只保存模型参数不够。
- [ ] 能区分 Model State 和 Optimizer State。
- [ ] 能独立写出保存 Checkpoint 的代码。
- [ ] 能独立写出加载 Checkpoint 的代码。
- [ ] 能理解 `start_epoch = checkpoint["epoch"] + 1`。
- [ ] 能亲自运行并敲一遍完整的 `checkpoint.py`。

我建议你今天尤其多敲几遍 `save_checkpoint()` 和 `load_checkpoint()` 这两个函数。它们是之后大模型训练工程中非常常见的代码。

**先把完整代码跑通，再提交上面的 5 道题。** 我会根据你的回答和代码运行情况判断 Day 21 是否真正掌握，再进入 Day 22。


---

# 后续学习记录：按原有水果分类流程重写 Day21

用户要求：以原来的水果分类训练代码为基础，增加 Day21 的保存和恢复功能。

可以！这次我们完全沿用你自己的水果分类代码，不重新换数据集、不换模型，也不改变你已经掌握的训练流程。

我们只在你原来的代码上增加 Day 21 的内容，让你清楚地看到：**Checkpoint 究竟应该插入训练流程的什么位置，以及如何在原有训练循环上实现保存和恢复。**

先说一下：你原来的训练代码整体没有问题，有两个小地方需要注意：

1. `torch.manual_seed=42` 应该改为 `torch.manual_seed(42)`，你原来是把函数覆盖成了数字，并没有设置随机种子。
2. 你的数据是 3 类，每类 100 个样本，因此 `x.shape` 实际是 `[300, 2]`，不是注释中的 `[900, 2]`。

# Day 21：在原有训练流程上加入 Checkpoint

## 一、先看今天究竟要改什么

你的原有训练流程：

**创建数据集和 DataLoader**

**创建模型、Loss、Optimizer**

**Forward → Loss → Backward → Optimizer.step()**

**统计 Loss 和 Accuracy**

**下一个 Epoch**

今天增加三个东西：

1. 创建 Scheduler 和 Scaler。
2. 每个 Epoch 训练结束后，保存 Checkpoint。
3. 模拟训练到 Epoch 5 后程序中断，重新创建模型并恢复状态，从 Epoch 6 继续训练。

## 今天的新训练流程

**Epoch 1–5：正常训练**

**保存 Checkpoint**

**模拟程序中断，重新启动**

**加载模型、Optimizer、Scheduler、Scaler**

**Epoch 6–10：继续训练**

## 二、完整代码 checkpoint.py

下面是完整可运行版本。我尽可能保留你的原始变量命名、注释和训练循环结构。

```python

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
# 5. 创建训练组件
# ============================================

def build_training_objects():

    model = FruitClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    # AMP 混合精度
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    return model, criterion, optimizer, scheduler, scaler

# ============================================
# 6. 训练参数
# ============================================

num_epochs = 10

checkpoint_path = "fruit_checkpoint.pth"

# ============================================
# 7. 训练一个 Epoch
# ============================================

def train_one_epoch(
    model,
    criterion,
    optimizer,
    scaler,
    global_step
):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_index, (batch_x, batch_y) in enumerate(loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. Forward + Loss
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda")
        ):

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

        # 3. Backward
        scaler.scale(loss).backward()

        # 4. Optimizer Step
        scaler.step(optimizer)

        scaler.update()

        # 5. Global Step
        global_step += 1

        # 6. 统计 Loss
        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size

        # 7. 统计 Accuracy
        predictions = torch.argmax(logits, dim=1)

        correct = (predictions == batch_y).sum().item()

        total_correct += correct

        total_samples += batch_size

        # 8. 打印部分 Batch
        if batch_index < 2:

            print(f"Batch {batch_index + 1}")

            print("Loss:", loss.item())

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy, global_step

# ============================================
# 8. 保存 Checkpoint
# ============================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step
):

    checkpoint = {

        # 模型参数
        "model_state_dict": model.state_dict(),

        # 优化器状态
        "optimizer_state_dict": optimizer.state_dict(),

        # 学习率调度器状态
        "scheduler_state_dict": scheduler.state_dict(),

        # AMP 状态
        "scaler_state_dict": scaler.state_dict(),

        # 当前 Epoch
        "epoch": epoch,

        # 总更新次数
        "global_step": global_step,

        # 训练配置
        "config": {
            "batch_size": 32,
            "learning_rate": 0.1,
            "num_epochs": num_epochs
        }
    }

    torch.save(checkpoint, checkpoint_path)

    print("Checkpoint 保存成功！")

# ============================================
# 9. 加载 Checkpoint
# ============================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # 恢复模型
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 恢复优化器
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 恢复 Scheduler
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # 恢复 Scaler
    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    # 恢复训练进度
    start_epoch = checkpoint["epoch"] + 1

    global_step = checkpoint["global_step"]

    print("Checkpoint 加载成功！")

    print("恢复 Epoch:", start_epoch + 1)

    print("恢复 Global Step:", global_step)

    return start_epoch, global_step

# ============================================
# 10. 第一次训练：Epoch 1-5
# ============================================

model, criterion, optimizer, scheduler, scaler = build_training_objects()

global_step = 0

for epoch in range(0, 5):

    print(f"\n========== Epoch {epoch + 1} ==========")

    avg_loss, accuracy, global_step = train_one_epoch(
        model,
        criterion,
        optimizer,
        scaler,
        global_step
    )

    # 更新学习率
    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%} "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # 每个 Epoch 结束后保存
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step
    )

# ============================================
# 11. 模拟程序中断
# ============================================

print("\n========== 模拟程序中断 ==========")

# ============================================
# 12. 重新创建模型和训练组件
# ============================================

model, criterion, optimizer, scheduler, scaler = build_training_objects()

# ============================================
# 13. 恢复 Checkpoint
# ============================================

start_epoch, global_step = load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler
)

# ============================================
# 14. 从 Epoch 6 继续训练
# ============================================

print("\n========== 恢复训练 ==========")

for epoch in range(start_epoch, num_epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    avg_loss, accuracy, global_step = train_one_epoch(
        model,
        criterion,
        optimizer,
        scaler,
        global_step
    )

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%} "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step
    )

print("\n训练全部完成！")
```

---

## 三、重点理解今天新增的代码

不需要重新学习你已经掌握的 Forward、Loss、Backward。我们重点攻克新增的部分。

### 1\. 为什么要单独封装 build\_training\_objects()？

```python
def build_training_objects():

    model = FruitClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    return model, criterion, optimizer, scheduler, scaler
```

因为我们现在有两次创建模型的需求：

- 第一次：开始训练。
- 第二次：模拟程序中断后，重新启动训练。

所以把它们封装起来，避免重复写创建代码。

特别注意：

**重新创建模型不等于恢复模型。**

重新创建时参数是重新初始化的，之后还要通过 `load_state_dict()` 恢复原来的参数。

### 2\. 为什么加入 Scheduler？

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5
)
```

它表示每经过 3 个 Epoch，学习率乘以 0.5。

今天主要是为了让你能够观察 Scheduler State 的保存与恢复。

你可以看到：

```python
scheduler.step()
```

在每个 Epoch 训练完成后执行。

而保存时：

```python
"scheduler_state_dict": scheduler.state_dict()
```

恢复时：

```python
scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)
```

### 3\. 今天最重要的两个函数

**保存：**

```python
def save_checkpoint(...):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }

    torch.save(checkpoint, checkpoint_path)
```

这里的核心就是：

把所有需要保存的状态放入一个字典，然后通过 `torch.save()` 写入文件。

**恢复：**

```python
def load_checkpoint(...):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
```

这就是今天最核心的工程代码。

### 4\. 为什么这里的 Epoch 要加 1？

我们保存的是 Python 循环中的 Epoch 索引。

例如：

```python
for epoch in range(0, 5):
```

实际对应：

| Python 中的 epoch | 实际训练轮次 |
|--------|--------|
| 0 | Epoch 1 |
| 1 | Epoch 2 |
| 2 | Epoch 3 |
| 3 | Epoch 4 |
| 4 | Epoch 5 |

假设保存时：

```python
"epoch": 4
```

表示 Epoch 5 已经完成。

所以：

```python
start_epoch = checkpoint["epoch"] + 1
```

得到 5。

然后：

```python
for epoch in range(start_epoch, num_epochs):
```

就会从索引 5 开始，也就是实际的 Epoch 6。

---

## 四、运行后重点观察什么？

**预期训练过程**

1. **Epoch 1–5**

   正常训练，每轮结束保存一次 Checkpoint。
2. **模拟中断**

   原来的模型、优化器等训练组件被重新创建。
3. **加载 Checkpoint**

   恢复模型参数、优化器状态、Scheduler 状态、Scaler 状态和训练进度。
4. **Epoch 6–10**

   接着训练，不再从 Epoch 1 开始。

另外，今天这份代码是 Epoch 边界的断点恢复。因为我们还没有保存随机数状态，所以恢复后的 DataLoader Shuffle 顺序不保证与中断前完全一致。这正是 Day 22 要解决的问题。

## 五、今天你需要真正掌握的东西

| 新增代码 | 你需要掌握什么 |
|--------|--------|
| `build_training_objects()` | 创建训练组件 |
| `scheduler.state_dict()` | 保存学习率调度状态 |
| `scaler.state_dict()` | 保存 AMP 状态 |
| `torch.save()` | 保存 Checkpoint |
| `torch.load()` | 读取 Checkpoint |
| `model.load_state_dict()` | 恢复模型参数 |
| `optimizer.load_state_dict()` | 恢复优化器状态 |
| `start_epoch` | 恢复训练位置 |

**一个小提醒：** 你当前代码没有使用验证集，所以这版没有额外引入 Validation，也没有保存 Best Model。我们只专注 Day 21 的 Checkpoint，不额外增加无关知识。

---

接下来你先做两件事：

1. 把完整代码自己敲一遍并运行，观察 Epoch 5 中断模拟后能否成功从 Epoch 6 继续。
2. 告诉我下面这个问题：

**假设 Epoch 5 训练结束后，我们保存了 Checkpoint。重新运行程序时，执行 `model = FruitClassifier()` 会重新初始化模型参数。那为什么后面执行 `model.load_state_dict(checkpoint["model_state_dict"])`，就能够恢复 Epoch 5 训练后的模型？**

这个问题答明白，说明你已经真正理解了模型结构创建和模型参数恢复之间的区别。


---

## 归档说明

保留完整初讲和后续按原训练流程重写的讲解，两套教学程序均保留。`code.py` 对应本地 `checkpoint.py`，归档时仅将权重输出位置改到脚本旁。Day21 演示恢复训练组件和 Epoch 进度，不承诺随机数和 Batch 顺序无缝接续，后者见 Day22。来源：[《学习模型检查点保存》](https://chatgpt.com/c/6afeab87-2860-83e9-a278-bbd28e8b44dc)。
