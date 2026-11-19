
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)
from contextlib import nullcontext


# ============================================
# 1. Day20：固定随机种子
# ============================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================
# 2. Day20：Debug 配置
# ============================================

debug_mode = True

# 可选：
# "normal"        正常训练
# "nan_input"     输入数据包含 NaN
# "bad_label"     标签越界
# "gradient_big"  人为放大梯度
# "gradient_zero" 人为制造零梯度

fault_mode = "nan_input"

# 是否执行单 Batch 过拟合测试
single_batch_test = False

# 是否开启 Autograd 异常检测
detect_anomaly = False


# ============================================
# 3. 设备与混合精度（Day19）
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"

if use_amp and torch.cuda.is_bf16_supported():

    amp_dtype = torch.bfloat16

else:

    amp_dtype = torch.float16

use_scaler = (
    use_amp and amp_dtype == torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_scaler
)

print("当前设备:", device)


# ============================================
# 4. 创建合成数据
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


# ============================================
# 5. Day20：人为制造数据异常
# ============================================

if fault_mode == "nan_input":

    x[0, 0] = float("nan")

elif fault_mode == "bad_label":

    y[0] = 5


# ============================================
# 6. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

# Day20：划分训练集和验证集

train_size = int(len(dataset) * 0.8)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================
# 7. 定义模型
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


model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# ============================================
# 8. 训练参数
# ============================================

num_epochs = 10

# Day17：梯度累积
accumulation_steps = 4

effective_batch_size = (
    train_loader.batch_size
    * accumulation_steps
)

# Day18：梯度裁剪
max_norm = 1.0


# ============================================
# 9. Day20：数据检查函数
# ============================================

def check_batch(batch_x, batch_y):

    assert batch_x.ndim == 2
    assert batch_x.shape[1] == 2

    assert batch_y.ndim == 1
    assert batch_x.size(0) == batch_y.size(0)

    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.long

    if not torch.isfinite(batch_x).all():
        raise RuntimeError("输入数据存在 NaN 或 Inf")

    if batch_y.min() < 0 or batch_y.max() >= 3:
        raise RuntimeError("标签超出类别范围")


# 在训练开始前检查整个数据集
if debug_mode:
    check_batch(x, y)


# ============================================
# 10. Day20：梯度检查函数
# ============================================

def check_gradients(model, print_details=False):

    has_gradient = False

    for name, param in model.named_parameters():

        if param.grad is None:

            if print_details:
                print(f"{name}: 没有梯度")

            continue

        has_gradient = True

        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"{name} 的梯度存在 NaN 或 Inf"
            )

        if print_details:

            grad_norm = param.grad.norm().item()

            print(
                f"{name}: "
                f"Grad Norm = {grad_norm:.8f}"
            )

    if not has_gradient:
        raise RuntimeError("所有模型参数均没有梯度")


# ============================================
# 11. Day20：单 Batch 过拟合测试
# ============================================

def overfit_one_batch():

    batch_x, batch_y = next(iter(train_loader))

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # 使用独立模型，避免影响正式训练
    test_model = FruitClassifier().to(device)

    test_optimizer = torch.optim.SGD(
        test_model.parameters(),
        lr=0.1
    )

    test_model.train()

    print("\n开始单 Batch 过拟合测试")

    # 使用普通 FP32 训练，减少干扰因素
    for step in range(200):

        test_optimizer.zero_grad()

        logits = test_model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        test_optimizer.step()

        if (step + 1) % 20 == 0:

            predictions = logits.argmax(dim=1)

            accuracy = (
                (predictions == batch_y)
                .float()
                .mean()
                .item()
            )

            print(
                f"Step {step + 1:3d} "
                f"Loss: {loss.item():.4f} "
                f"Accuracy: {accuracy:.2%}"
            )


# ============================================
# 12. Day20：验证集评估函数
# ============================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================
# 13. 可选：单 Batch 过拟合测试
# ============================================

if single_batch_test:

    overfit_one_batch()


# ============================================
# 14. 完整训练循环
# ============================================

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_index, (batch_x, batch_y) in enumerate(
        train_loader
    ):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Day20：检查输入数据
        if debug_mode:
            check_batch(batch_x, batch_y)

        # 根据配置选择是否开启异常检测
        context = (
            torch.autograd.detect_anomaly()
            if detect_anomaly
            else nullcontext()
        )

        with context:

            # Day19：自动混合精度
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):

                logits = model(batch_x)

                # Day20：检查 Logits
                if debug_mode:

                    if not torch.isfinite(logits).all():
                        raise RuntimeError(
                            "Logits 存在 NaN 或 Inf"
                        )

                loss = criterion(logits, batch_y)

            # Day20：检查 Loss
            if debug_mode:

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Loss 存在 NaN 或 Inf"
                    )

            # =================================
            # Day17：梯度累积
            # =================================

            batch_size = batch_x.size(0)

            group_index = (
                batch_index // accumulation_steps
            )

            group_start = (
                group_index * effective_batch_size
            )

            group_size = min(
                effective_batch_size,
                len(train_dataset) - group_start
            )

            scaled_loss = (
                loss * batch_size / group_size
            )

            # =================================
            # Day20：人为制造梯度异常
            # =================================

            if fault_mode == "gradient_big":

                scaled_loss = scaled_loss * 10000

            elif fault_mode == "gradient_zero":

                scaled_loss = scaled_loss * 0.0

            # Day19：混合精度反向传播
            scaler.scale(scaled_loss).backward()

        # 判断是否完成一个累积周期
        is_full_cycle = (
            (batch_index + 1)
            % accumulation_steps == 0
        )

        is_last_batch = (
            batch_index + 1 == len(train_loader)
        )

        if is_full_cycle or is_last_batch:

            # Day19：还原梯度
            scaler.unscale_(optimizer)

            # Day20：检查每一层的真实梯度
            if debug_mode:

                check_gradients(
                    model,
                    print_details=(epoch == 0)
                )

            # Day18：梯度裁剪
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
                error_if_nonfinite=debug_mode
            ).item()

            # Day20：梯度异常监控
            if debug_mode:

                if grad_norm > 100:

                    print(
                        f"警告：梯度范数过大 "
                        f"{grad_norm:.6f}"
                    )

                elif grad_norm < 1e-8:

                    print(
                        f"警告：梯度范数极小 "
                        f"{grad_norm:.10f}"
                    )

            # Day19：更新参数
            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad()

        # =====================================
        # 统计训练指标
        # =====================================

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_samples += batch_size

    # =========================================
    # 15. 计算训练指标
    # =========================================

    train_loss = total_loss / total_samples

    train_accuracy = (
        total_correct / total_samples
    )

    # Day20：计算验证集指标
    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}] "
        f"\nTrain Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2%} "
        f"\nVal Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2%}"
    )


print("\n训练完成！")
