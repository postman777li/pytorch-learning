import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# ============================================
# 1. 创建数据（沿用 train_loop.py）
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

x = torch.cat([class_0, class_1, class_2], dim=0)

y = torch.cat([
    torch.zeros(num_per_class, dtype=torch.long),
    torch.ones(num_per_class, dtype=torch.long),
    torch.full((num_per_class,), 2, dtype=torch.long)
])

dataset = TensorDataset(x, y)

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

# ============================================
# 3. 定义训练函数
# ============================================

def train_model(model, optimizer, loader, epochs=20):

    criterion = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_samples = 0

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            loss.backward()

            optimizer.step()

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)

        print(
            f"Epoch {epoch + 1:02d} "
            f"Loss: {avg_loss:.4f}"
        )

    return loss_history

# ============================================
# 4. 定义三种优化器配置
# ============================================

configs = {
    "SGD": {
        "type": "sgd",
        "lr": 0.01
    },
    "Adam": {
        "type": "adam",
        "lr": 0.001
    },
    "AdamW": {
        "type": "adamw",
        "lr": 0.001
    }
}

# ============================================
# 5. 保存相同的初始模型参数
# ============================================

initial_model = FruitClassifier()
initial_weights = {
    key: value.clone()
    for key, value in initial_model.state_dict().items()
}

results = {}

# ============================================
# 6. 分别训练并比较
# ============================================

for name, config in configs.items():

    print(f"\n========== {name} ==========")

    model = FruitClassifier()
    model.load_state_dict(initial_weights)

    if config["type"] == "sgd":

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"]
        )

    elif config["type"] == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.01
        )

    else:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.01
        )

    # 为每个优化器创建相同的 Batch 顺序
    generator = torch.Generator()
    generator.manual_seed(123)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=generator
    )

    loss_history = train_model(
        model,
        optimizer,
        loader,
        epochs=20
    )

    results[name] = loss_history

# ============================================
# 7. 打印对比结果
# ============================================

print("\n========== 最终对比 ==========")

for name, history in results.items():

    print(
        f"{name:6s} "
        f"初始 Loss: {history[0]:.4f} "
        f"最终 Loss: {history[-1]:.4f}"
    )
