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