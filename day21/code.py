
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

from pathlib import Path

checkpoint_path = str(Path(__file__).with_name("fruit_checkpoint.pth"))


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
