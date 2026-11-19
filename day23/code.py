
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split

# ============================================
# 1. 基础配置
# ============================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_classes = 3

num_epochs = 20

batch_size = 32

from pathlib import Path

best_model_path = str(Path(__file__).with_name("best_model.pth"))

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
# 3. 创建 Dataset
# ============================================

dataset = TensorDataset(x, y)

print("样本总数:", len(dataset))

# ============================================
# 4. 划分 Train / Validation / Test
# ============================================

train_size = int(0.7 * len(dataset))

val_size = int(0.15 * len(dataset))

test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(

    dataset,

    [train_size, val_size, test_size],

    generator=torch.Generator().manual_seed(42)
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# ============================================
# 5. 创建三个 DataLoader
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ============================================
# 6. 定义分类模型
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
# 7. 创建模型、Loss、Optimizer
# ============================================

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

# ============================================
# 8. 训练一个 Epoch
# ============================================

def train_one_epoch():

    model.train()

    total_loss = 0

    total_correct = 0

    total_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. Forward
        logits = model(batch_x)

        # 3. Loss
        loss = criterion(logits, batch_y)

        # 4. Backward
        loss.backward()

        # 5. Optimizer
        optimizer.step()

        # 6. 统计 Loss
        batch_size_now = batch_x.size(0)

        total_loss += loss.item() * batch_size_now

        # 7. 统计 Accuracy
        predictions = torch.argmax(logits, dim=1)

        correct = (
            predictions == batch_y
        ).sum().item()

        total_correct += correct

        total_samples += batch_size_now

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy

# ============================================
# 9. 评估函数
# ============================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0

    total_samples = 0

    # 混淆矩阵
    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)

            batch_y = batch_y.to(device)

            # 1. Forward
            logits = model(batch_x)

            # 2. Loss
            loss = criterion(logits, batch_y)

            batch_size_now = batch_x.size(0)

            total_loss += loss.item() * batch_size_now

            total_samples += batch_size_now

            # 3. 预测类别
            predictions = torch.argmax(logits, dim=1)

            # 4. 构建当前 Batch 的混淆矩阵
            encoded = (
                batch_y.cpu() * num_classes
                + predictions.cpu()
            )

            batch_cm = torch.bincount(
                encoded,
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

            # 5. 累加混淆矩阵
            confusion_matrix += batch_cm

    # ========================================
    # 10. 计算评估指标
    # ========================================

    avg_loss = total_loss / total_samples

    # 对角线：每个类别预测正确的数量
    tp = confusion_matrix.diag().float()

    # 每行求和：每个类别实际的数量
    actual = confusion_matrix.sum(dim=1).float()

    # 每列求和：每个类别被预测的数量
    predicted = confusion_matrix.sum(dim=0).float()

    # Precision
    precision = tp / predicted.clamp(min=1)

    # Recall
    recall = tp / actual.clamp(min=1)

    # F1
    f1 = 2 * precision * recall / (
        precision + recall
    ).clamp(min=1e-12)

    # Accuracy
    accuracy = tp.sum() / confusion_matrix.sum()

    # Macro Average
    macro_precision = precision.mean()

    macro_recall = recall.mean()

    macro_f1 = f1.mean()

    # Weighted F1
    weighted_f1 = (
        f1 * actual
    ).sum() / actual.sum()

    return {

        "loss": avg_loss,

        "accuracy": accuracy.item(),

        "precision": macro_precision.item(),

        "recall": macro_recall.item(),

        "f1": macro_f1.item(),

        "weighted_f1": weighted_f1.item(),

        "class_precision": precision.tolist(),

        "class_recall": recall.tolist(),

        "class_f1": f1.tolist(),

        "confusion_matrix": confusion_matrix
    }

# ============================================
# 11. 训练与 Validation
# ============================================

best_val_loss = float("inf")

best_epoch = 0

class_names = ["苹果", "香蕉", "橙子"]

for epoch in range(num_epochs):

    # 1. Training
    train_loss, train_accuracy = train_one_epoch()

    # 2. Validation
    val_metrics = evaluate(model, val_loader)

    # 3. Learning Rate
    current_lr = optimizer.param_groups[0]["lr"]

    # 4. 更新 Scheduler
    scheduler.step()

    print(f"\n========== Epoch {epoch + 1} ==========")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%}"
    )

    print(
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.2%}"
    )

    print(
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    print(f"Learning Rate: {current_lr:.6f}")

    # 5. 保存最佳模型
    if val_metrics["loss"] < best_val_loss:

        best_val_loss = val_metrics["loss"]

        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_loss": best_val_loss
            },
            best_model_path
        )

        print("发现更好的模型，已保存！")

# ============================================
# 12. 加载最佳模型
# ============================================

print("\n========== 加载最佳模型 ==========")

best_checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

print("最佳 Epoch:", best_checkpoint["epoch"])

print("最佳 Val Loss:", best_checkpoint["val_loss"])

# ============================================
# 13. 最终测试（归档补全，非取得的窗口原文）
# ============================================
# 仅在验证集选定并加载最佳权重之后评估测试集。
test_metrics = evaluate(model, test_loader)

print("\n========== 最终 Test Evaluation ==========")
for key in ("loss", "accuracy", "precision", "recall", "f1", "weighted_f1"):
    print(f"{key}: {test_metrics[key]:.6f}")

print("混淆矩阵（行是真实类别，列是预测类别）:")
print(test_metrics["confusion_matrix"])

for index, name in enumerate(class_names):
    print(
        f"{name}: Precision={test_metrics['class_precision'][index]:.4f} "
        f"Recall={test_metrics['class_recall'][index]:.4f} "
        f"F1={test_metrics['class_f1'][index]:.4f}"
    )
