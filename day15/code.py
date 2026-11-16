import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# 1. 数据：沿用之前的三分类任务
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

# 2. 模型：沿用 FruitClassifier
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

# 3. 保存相同的初始权重
initial_model = FruitClassifier()
initial_weights = {
    k: v.clone()
    for k, v in initial_model.state_dict().items()
}

# 4. 定义训练函数
def train_model(lr, epochs=30):

    model = FruitClassifier()
    model.load_state_dict(initial_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0
    )

    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(123)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=generator
    )

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

    return loss_history

# 5. 对比不同学习率
learning_rates = [0.0001, 0.001, 0.01, 0.1]

results = {}

for lr in learning_rates:

    history = train_model(lr)

    results[lr] = history

    print(
        f"lr={lr:<8} "
        f"最终 Loss={history[-1]:.4f}"
    )

# 6. 绘制 Loss 曲线
for lr, history in results.items():

    plt.plot(
        history,
        label=f"lr={lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.show()
