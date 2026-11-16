import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

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


# t = (1, 2, 3)        # 三个元素的元组
# t = (1, "hello", 3.5) # 元素类型可以混着来
# t = ()               # 空元组
# t = (5,)             # 只有一个元素的元组（注意逗号！）

# ============================================
# 3. Dataset 和 DataLoader
# ============================================


# dataset = [
#     (x[0], y[0]),   →  ( [1.1, 0.9], 0 )
#     (x[1], y[1]),   →  ( [0.8, 1.2], 0 )
#     (x[2], y[2]),   →  ( [1.0, 1.0], 0 )
#     (x[3], y[3]),   →  ( [2.9, 1.1], 1 )
#     (x[4], y[4]),   →  ( [3.1, 0.8], 1 )
#     (x[5], y[5]),   →  ( [3.0, 1.0], 1 )
#     (x[6], y[6]),   →  ( [2.0, 3.1], 2 )
#     (x[7], y[7]),   →  ( [1.9, 2.8], 2 )
#     (x[8], y[8]),   →  ( [2.1, 3.0], 2 )
# ]
dataset=TensorDataset(x,y)

loader=DataLoader(
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

        self.network=nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self,x):
        return self.network(x)

# ============================================
# 5. 创建模型、损失函数和优化器
# ============================================
model=FruitClassifier()

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# ============================================
# 6. 训练参数
# ============================================

num_epochs=10

# ============================================
# 7. 完整训练循环
# ============================================
for epoch in range(num_epochs):
    # 切换到训练模式
    model.train()
    # 每个 Epoch 开始时清零统计指标
    total_loss=0
    total_correct=0
    total_samples=0

    # 遍历当前 Epoch 的所有 Batch
    #加上enumerate
    # (0, (batch_x, batch_y))
    #(1, (batch_x, batch_y))  x是一个batch的样本 有32个，y是标签
    for batch_index,(batch_x,batch_y) in enumerate(loader):

        # 1. 清零旧梯度
        optimizer.zero_grad()

        # 2. Forward
        logits=model(batch_x)

        # 3. Calculate Loss
        loss=criterion(logits,batch_y)

        ## 4. Backward
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

        # 6. 统计当前 Batch 的样本数量
        batch_size=batch_x.size(0)

        # 7. 累计 Loss
        total_loss+=loss.item()*batch_size

        # 8. 得到预测类别
        predictions=torch.argmax(logits,dim=1)

        # 9. 累计预测正确数量
        correct=(predictions==batch_y).sum().item()
        total_correct += correct

        # 10. 累计样本数量
        total_samples += batch_size

        # 11. 第一轮打印部分 Batch 信息
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
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2%}"
    )

# ============================================
# 9. 训练结束
# ============================================

print("\n训练完成！")