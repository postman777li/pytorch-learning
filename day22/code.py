
import os
import json
import logging

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================
# 1. 实验配置
# ============================================

config = {
    "seed": 42,
    "num_per_class": 100,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "scheduler_step_size": 3,
    "scheduler_gamma": 0.5
}

from pathlib import Path

experiment_dir = str(Path(__file__).resolve().parent / "runs" / "fruit_exp01")

# 用于模拟训练中断
# 第一次运行设置为5
# 第二次运行修改为None
stop_after_epoch = 5


# ============================================
# 2. 创建实验目录
# ============================================

os.makedirs(experiment_dir, exist_ok=True)

checkpoint_path = os.path.join(
    experiment_dir, "checkpoint.pth"
)

config_path = os.path.join(
    experiment_dir, "config.json"
)

log_path = os.path.join(
    experiment_dir, "train.log"
)

tensorboard_dir = os.path.join(
    experiment_dir, "tensorboard"
)


# ============================================
# 3. 保存或检查实验配置
# ============================================

if os.path.exists(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        old_config = json.load(f)

    if old_config != config:
        raise ValueError(
            "实验配置发生变化，请使用新的实验目录"
        )

else:

    with open(config_path, "w", encoding="utf-8") as f:

        json.dump(config, f, indent=4)


# ============================================
# 4. 初始化 Logging
# ============================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

    handlers=[
        logging.FileHandler(
            log_path,
            mode="a",
            encoding="utf-8"
        ),

        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================
# 5. 随机种子与设备
# ============================================

torch.manual_seed(config["seed"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"

logger.info(f"当前设备: {device}")


# ============================================
# 6. 创建合成训练数据
# ============================================

num_per_class = config["num_per_class"]

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
# 7. Dataset 和 DataLoader
# ============================================

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=0
)

logger.info(f"样本总数: {len(dataset)}")

logger.info(f"每轮Batch数量: {len(loader)}")


# ============================================
# 8. 定义分类模型
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
# 9. 创建训练组件
# ============================================

model = FruitClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["learning_rate"],
    momentum=config["momentum"]
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["scheduler_step_size"],
    gamma=config["scheduler_gamma"]
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================
# 10. 训练一个 Epoch
# ============================================

def train_one_epoch(global_step):

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
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp
        ):

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

        # 3. Backward
        scaler.scale(loss).backward()

        # 4. Optimizer Step
        old_scale = scaler.get_scale()

        scaler.step(optimizer)

        scaler.update()

        # 仅统计未因AMP溢出而跳过的更新
        if scaler.get_scale() >= old_scale:

            global_step += 1

        # 5. 统计 Loss
        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size

        # 6. 统计 Accuracy
        predictions = torch.argmax(logits, dim=1)

        correct = (
            predictions == batch_y
        ).sum().item()

        total_correct += correct

        total_samples += batch_size

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy, global_step


# ============================================
# 11. 保存 Checkpoint
# ============================================

def save_checkpoint(epoch, global_step):

    checkpoint = {

        # Day 21：训练状态
        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "epoch": epoch,

        "global_step": global_step,

        "config": config,

        "device_type": device.type,

        # Day 22：随机数状态
        "cpu_rng_state": torch.get_rng_state(),

        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if use_amp
            else None
        )
    }

    # 先保存到临时文件
    temp_path = checkpoint_path + ".tmp"

    torch.save(checkpoint, temp_path)

    # 保存成功后替换旧文件
    os.replace(temp_path, checkpoint_path)

    logger.info("Checkpoint 保存成功")


# ============================================
# 12. 加载 Checkpoint
# ============================================

def load_checkpoint():

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["config"] != config:

        raise ValueError("Checkpoint 配置不一致")

    if checkpoint["device_type"] != device.type:

        raise ValueError("续训设备类型发生变化")

    # 恢复模型参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 恢复优化器状态
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 恢复Scheduler状态
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # 恢复AMP状态
    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    # 恢复训练进度
    start_epoch = checkpoint["epoch"] + 1

    global_step = checkpoint["global_step"]

    # 恢复CPU随机数状态
    torch.set_rng_state(
        checkpoint["cpu_rng_state"]
    )

    # 恢复CUDA随机数状态
    if use_amp:

        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )

    return start_epoch, global_step


# ============================================
# 13. 判断是否需要断点续训
# ============================================

start_epoch = 0

global_step = 0

is_resume = os.path.exists(checkpoint_path)

if is_resume:

    start_epoch, global_step = load_checkpoint()

    logger.info("Checkpoint 加载成功")

    logger.info(f"已完成Epoch: {start_epoch}")

    logger.info(f"恢复Global Step: {global_step}")

else:

    logger.info("未发现Checkpoint，从头开始训练")


# ============================================
# 14. 创建 TensorBoard Writer
# ============================================

writer = SummaryWriter(
    log_dir=tensorboard_dir,

    purge_step=(
        start_epoch + 1
        if is_resume
        else None
    )
)


# ============================================
# 15. 完整训练循环
# ============================================

num_epochs = config["num_epochs"]

try:

    for epoch in range(start_epoch, num_epochs):

        # 记录当前Epoch实际使用的学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 训练一个Epoch
        avg_loss, accuracy, global_step = train_one_epoch(
            global_step
        )

        # 更新下一轮使用的学习率
        scheduler.step()

        # ------------------------------------
        # 1. 记录文本日志
        # ------------------------------------

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"LR: {current_lr:.6f} | "
            f"Global Step: {global_step}"
        )

        # ------------------------------------
        # 2. 记录TensorBoard
        # ------------------------------------

        writer.add_scalar(
            "Loss/train",
            avg_loss,
            epoch + 1
        )

        writer.add_scalar(
            "Accuracy/train",
            accuracy,
            epoch + 1
        )

        writer.add_scalar(
            "LearningRate",
            current_lr,
            epoch + 1
        )

        writer.flush()

        # ------------------------------------
        # 3. 保存Checkpoint
        # ------------------------------------

        save_checkpoint(epoch, global_step)

        # ------------------------------------
        # 4. 模拟程序中断
        # ------------------------------------

        if stop_after_epoch == epoch + 1:

            logger.warning("模拟训练中断，退出训练循环")

            break

    else:

        logger.info("全部训练完成！")

except KeyboardInterrupt:

    logger.warning(
        "用户中断训练，下次从最近的Checkpoint恢复"
    )

finally:

    writer.close()
