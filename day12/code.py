import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader,random_split
import copy
from pathlib import Path
# ============================================
# 1. 固定随机种子
# ============================================

torch.manual_seed(42)

# ============================================
# 2. 创建合成训练数据
# ============================================

num_per_class=100

class_0=(
    #100个数据 两个特征
    torch.randn(num_per_class,2)*0.25+torch.tensor([1.0,1.0])
)

class_1=(
    torch.randn(num_per_class,2)*0.25+torch.tensor([3.0,1.0])
)

class_2=(
    torch.randn(num_per_class,2)*0.25+torch.tensor([2.0,3.0])
)

# x.shape 是 [300, 2]
x=torch.cat(
    [class_0,class_1,class_2],
    #dim=0 → 沿行方向堆叠（竖直往下拼），形状变化是：行数相加，列数不变。
    #dim=1 → 沿列方向拼（水平并排），行数不变，列数相加。
    dim=0
)

# 生成对应标签 dim默认是0 竖着排
y=torch.cat([
    torch.zeros(num_per_class,dtype=torch.long),
    torch.ones(num_per_class, dtype=torch.long),
    #(num_per_class,)是一维  (num_per_class,2)是2维
    #元素的值是2 第一个参数是size 如果直接(100)这只是数字 和100没区别 而(100,)则是一维
    torch.full((num_per_class,),2,dtype=torch.long)
])

print("X shape:", x.shape)
print("y shape:", y.shape)

dataset=TensorDataset(x,y)

# ============================================
# 3. 划分训练、验证和测试数据
# ============================================

train_dataset,val_dataset,test_dataset=random_split(
    dataset,
    [210,60,30],
    generator=torch.Generator().manual_seed(42)
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

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================
# 4. 定义模型
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
# 6. 定义评估函数
# ============================================
def evaluate(model,loader,criterion):
    model.eval()

    total_loss=0
    total_correct=0
    total_samples=0

    with torch.no_grad():
        for batch_x,batch_y in loader:

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )
            #t = tensor(3)
            # t.item()        # 3   ← 普通 Python int
            # type(t.item())  # <class 'int'>
            total_correct += (
                predictions == batch_y
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy

# ============================================
# 7. Early Stopping 配置
# ============================================

num_epochs = 50

patience = 5

min_delta = 0.0001

best_val_loss = float("inf")

best_model_state = None

best_epoch = 0

epochs_without_improvement = 0

# ============================================
# 8. 完整训练与验证循环
# ============================================

for epoch in range(num_epochs):

    # 切换训练模式
    model.train()

    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    # ---------- Train ----------

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        total_train_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_train_correct += (
            predictions == batch_y
        ).sum().item()

        total_train_samples += batch_size

    train_loss = (
        total_train_loss / total_train_samples
    )

    train_acc = (
        total_train_correct / total_train_samples
    )

    # ---------- Validation ----------

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion
    )

    # ---------- 打印指标 ----------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.2%}"
    )

    # ---------- Early Stopping ----------

    if val_loss < best_val_loss - min_delta:

        best_val_loss = val_loss

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print("\nEarly Stopping triggered!")
            break
# ============================================
# 9. 恢复最佳模型参数
# ============================================

model.load_state_dict(best_model_state)

print("\nBest Epoch:", best_epoch)

print("Best Val Loss:", best_val_loss)


# ============================================
# 10. 在测试集上进行最终评估
# ============================================

test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion
)

print("\n========== Final Test ==========")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")


# ============================================
# 11. 保存最佳模型
# ============================================

torch.save(
    model.state_dict(),
    Path(__file__).with_name("best_fruit_model.pth")
)

print("\n最佳模型已保存！")