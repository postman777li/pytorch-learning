import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ============================================
# 1. 创建三分类数据
# ============================================

num_per_class = 100

class_0 = torch.randn(100, 2) * 0.25 + torch.tensor([1., 1.])
class_1 = torch.randn(100, 2) * 0.25 + torch.tensor([3., 1.])
class_2 = torch.randn(100, 2) * 0.25 + torch.tensor([2., 3.])

x = torch.cat([class_0, class_1, class_2], dim=0)

y = torch.cat([
    torch.zeros(100, dtype=torch.long),
    torch.ones(100, dtype=torch.long),
    torch.full((100,), 2, dtype=torch.long)
])

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ============================================
# 2. 定义模型
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

model = FruitClassifier()

criterion = nn.CrossEntropyLoss()

# ============================================
# 3. 定义 AdamW 优化器
# ============================================

max_lr = 0.01

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    weight_decay=0.01
)

# ============================================
# 4. 定义训练步数
# ============================================

num_epochs = 30

total_steps = num_epochs * len(loader)

warmup_steps = int(total_steps * 0.1)

warmup_steps = max(1, warmup_steps)

min_lr_ratio = 0.1

# ============================================
# 5. 定义 Warmup + Cosine
# ============================================

def lr_lambda(step):

    # Warmup 阶段
    if step < warmup_steps:

        return (
            (step + 1) / warmup_steps
        )

    # Cosine 阶段
    progress = (
        step - warmup_steps + 1
    ) / max(1, total_steps - warmup_steps)

    progress = min(1.0, progress)

    cosine_ratio = 0.5 * (
        1 + math.cos(math.pi * progress)
    )

    return (
        min_lr_ratio
        + (1 - min_lr_ratio) * cosine_ratio
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda
)

# ============================================
# 6. 记录训练数据
# ============================================

lr_history = []
loss_history = []

# ============================================
# 7. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_samples = 0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        # Forward
        logits = model(batch_x)

        # Loss
        loss = criterion(logits, batch_y)

        # Backward
        loss.backward()

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 更新模型参数
        optimizer.step()

        # 更新下一步学习率
        scheduler.step()

        # 保存学习率
        lr_history.append(current_lr)

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples

    loss_history.append(avg_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"LR: {current_lr:.8f}"
    )

# ============================================
# 8. 绘制 Learning Rate 曲线
# ============================================

plt.figure()

plt.plot(lr_history)

plt.xlabel("Optimizer Step")
plt.ylabel("Learning Rate")

plt.title("Warmup + Cosine Decay")

plt.grid(True)

plt.show()

# ============================================
# 9. 绘制 Loss 曲线
# ============================================

plt.figure()

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title("Training Loss")

plt.grid(True)

plt.show()
