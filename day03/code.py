# 来源：《学习Day1张量》Day3 第八部分教学代码；非本人独立练习记录。
# 补充第1–7部分的短例子；权重路径改为本脚本旁，避免随工作目录变化。
import torch
import torch.nn as nn

from pathlib import Path


def concept_examples():
    """对应笔记第1–7部分；示例补充整理自学习会话。"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    simple = SimpleModel()
    print("Module / forward:", simple(torch.tensor([[2.0, 3.0]])).shape)

    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[4.0, 5.0]]))
        linear.bias.fill_(1.0)
    print("Linear 手算验证:", linear(torch.tensor([[2.0, 3.0]])).item())
    print("Linear 三维输入:", nn.Linear(512, 128)(torch.randn(2, 10, 512)).shape)

    class ParameterExample(nn.Module):
        def __init__(self):
            super().__init__()
            self.unregistered = torch.tensor(3.0, requires_grad=True)
            self.w = nn.Parameter(torch.tensor(3.0))

        def forward(self, x):
            return self.w * x

    example = ParameterExample()
    print("Parameter 注册:", list(dict(example.named_parameters())))
    print("Parameter 前向:", example(torch.tensor(2.0)).item())

    network = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    print("参数名称和形状:")
    for name, param in network.named_parameters():
        print(name, param.shape)
    print("参数总数:", sum(p.numel() for p in network.parameters()))  # 17
    print("Sequential 输出:", network(torch.randn(3, 2)).shape)
    print("ReLU:", nn.ReLU()(torch.tensor([-2.0, 0.0, 3.0])))
    print("state_dict 键:", list(network.state_dict()))


concept_examples()

# 以下保留学习会话第八部分完整训练流程。

# ===============================
# 1. 固定随机种子
# ===============================

torch.manual_seed(42)

# ===============================
# 2. 定义模型
# ===============================

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# 3. 创建模型
# ===============================

model = MyModel()

print("模型结构:")
print(model)

print("\n参数信息:")
for name, param in model.named_parameters():
    print(name, param.shape)

# ===============================
# 4. 构造训练数据
# ===============================

X = torch.rand(200, 2)

# 真实关系 y = 2*x1 + 3*x2 + 1
y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1

# ===============================
# 5. 定义损失函数和优化器
# ===============================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ===============================
# 6. 训练模型
# ===============================

for epoch in range(500):

    # 前向传播
    pred = model(X)

    # 计算损失
    loss = loss_fn(pred, y)

    # 清空旧梯度
    optimizer.zero_grad()

    # 计算梯度
    loss.backward()

    # 更新模型参数
    optimizer.step()

    # 每50轮输出一次
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1}, "
            f"Loss: {loss.item():.6f}"
        )

# ===============================
# 7. 测试模型
# ===============================

test_x = torch.tensor([[0.5, 0.8]])

with torch.no_grad():
    test_pred = model(test_x)

true_y = 2 * 0.5 + 3 * 0.8 + 1

print("\n测试预测:", test_pred.item())
print("真实答案:", true_y)

# ===============================
# 8. 保存权重
# ===============================

torch.save(
    model.state_dict(),
    Path(__file__).with_name("model_weights.pth")
)

print("模型权重保存完成")

# ===============================
# 9. 加载权重
# ===============================

new_model = MyModel()

state = torch.load(
    Path(__file__).with_name("model_weights.pth"),
    map_location="cpu",
    weights_only=True
)

new_model.load_state_dict(state)

with torch.no_grad():
    loaded_pred = new_model(test_x)

print("加载后预测:", loaded_pred.item())
