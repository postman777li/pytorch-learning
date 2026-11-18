
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

# ============================================
# 2. 选择设备和混合精度类型
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 判断是否使用 CUDA
use_amp = device.type == "cuda"

# 如果 GPU 支持 BF16，优先使用 BF16
if use_amp and torch.cuda.is_bf16_supported():

    amp_dtype = torch.bfloat16
    precision_name = "BF16"

else:

    amp_dtype = torch.float16
    precision_name = "FP16"


use_scaler = (
    use_amp and amp_dtype == torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_scaler
)

print("当前设备:", device)

if use_amp:
    print("当前混合精度:", precision_name)
else:
    print("当前使用 FP32")

print("GradScaler 是否启用:", use_scaler)

# ============================================
# 3. 创建合成训练数据
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
# 4. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ============================================
# 5. 定义模型
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
# 6. 模型、损失函数、优化器
# ============================================

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# ============================================
# 7. 训练参数
# ============================================

num_epochs = 10

# Day17：梯度累积
accumulation_steps = 4

micro_batch_size = loader.batch_size

effective_batch_size = (
    micro_batch_size * accumulation_steps
)

# Day18：梯度裁剪
max_norm = 1.0

# ============================================
# 8. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_index, (batch_x, batch_y) in enumerate(loader):

        # 将数据移动到 GPU 或 CPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # ====================================
        # Day19：autocast 自动混合精度
        # ====================================

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):

            # 1. Forward
            logits = model(batch_x)

            # 2. Calculate Loss
            loss = criterion(logits, batch_y)

        # ====================================
        # Day17：梯度累积
        # ====================================

        batch_size = batch_x.size(0)

        group_index = batch_index // accumulation_steps

        group_start = group_index * effective_batch_size

        group_size = min(
            effective_batch_size,
            len(dataset) - group_start
        )

        scaled_loss = loss * batch_size / group_size

        # ====================================
        # Day19：GradScaler
        # ====================================

        # FP16 时放大 Loss
        # BF16 / FP32 时自动退化为普通 backward
        scaler.scale(scaled_loss).backward()

        # 判断是否更新参数
        is_full_cycle = (
            (batch_index + 1) % accumulation_steps == 0
        )

        is_last_batch = (
            (batch_index + 1) == len(loader)
        )

        if is_full_cycle or is_last_batch:

            # =================================
            # Day19：还原梯度
            # =================================

            scaler.unscale_(optimizer)

            # =================================
            # Day18：梯度裁剪
            # =================================

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm
            )

            # =================================
            # Day19：更新参数
            # =================================

            scaler.step(optimizer)

            # 更新 GradScaler 的缩放因子
            scaler.update()

            # 清零梯度
            optimizer.zero_grad()

            if epoch == 0:
                print(
                    f"Batch {batch_index + 1} "
                    f"Grad Norm: {grad_norm.item():.4f}"
                )

        # ====================================
        # 9. 统计训练指标
        # ====================================

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        correct = (predictions == batch_y).sum().item()

        total_correct += correct

        total_samples += batch_size

    # ========================================
    # 10. 计算 Epoch 指标
    # ========================================

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )

# ============================================
# 11. 训练结束
# ============================================

print("\n训练完成！")
