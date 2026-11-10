# 来源：本地 device_and_mode.py；原文件保留。
# 整理改动：修正两处 is_available() 调用；权重保存到本脚本旁。
import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np



print(torch.cuda.is_available())
print(torch.cuda.device_count())

torch.manual_seed(42)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


#device=torch.device("cuda")
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#将 Tensor 移动到指定设备
x=torch.randn(2,3)
x=x.to(device)
print(x.device)

#定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(
            torch.tensor([2.0])
        )
        #注册之后，PyTorch 才能统一管理它：
        # - 跟随模型移动设备。
        # - 默认保存到 state_dict()。
        # - 不会被当成普通可训练参数交给优化器
        self.register_buffer(
            "running_value",
            torch.tensor([10.0])
        )
        self.net=nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.net(x)
#模型也需要移动
model=MyModel()
model=model.to(device)


#train()和eval()
dropout=nn.Dropout(p=0.5)
x=torch.ones(10)

# 训练模式
dropout.train()
print(dropout(x))

# 推理模式
dropout.eval()
print(dropout(x))



print(list(model.named_parameters()))
print(list(model.named_buffers()))
print(model.state_dict())










# ==================================
# 1. 设置随机种子
# ==================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ==================================
# 2. 选择设备
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)


# ==================================
# 3. 定义模型
# ==================================

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ==================================
# 4. 创建模型并移动设备
# ==================================

model = MyModel().to(device)

print("模型设备:", next(model.parameters()).device)


# ==================================
# 5. 创建训练数据并移动设备
# ==================================

X = torch.rand(200, 2)
y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1

X = X.to(device)
y = y.to(device)


# ==================================
# 6. 定义损失函数和优化器
# ==================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==================================
# 7. 训练模型
# ==================================

model.train()

for epoch in range(200):

    pred = model(X)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )


# ==================================
# 8. 评估模型
# ==================================

model.eval()

test_x = torch.tensor(
    [[0.5, 0.8]]
).to(device)

with torch.no_grad():

    test_pred = model(test_x)

print("预测结果:", test_pred.item())
print("真实结果:", 4.4)


# ==================================
# 9. 查看模型参数和缓冲区
# ==================================

print("\n模型参数:")

for name, param in model.named_parameters():
    print(name, param.shape, param.device)

print("\n模型缓冲区:")

for name, buffer in model.named_buffers():
    print(name, buffer.shape, buffer.device)


# ==================================
# 10. 保存模型权重
# ==================================

torch.save(
    model.state_dict(),
    Path(__file__).with_name("model_weights.pth")
)

print("\n模型权重保存完成")
