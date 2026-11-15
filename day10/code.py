
import torch
import torch.nn as nn


# =====================================
# 1. 固定随机种子
# =====================================

torch.manual_seed(42)


# =====================================
# 2. 准备数据
# =====================================

x = torch.tensor([
    [0.8, 0.6],
    [0.9, 0.3],
    [0.5, 0.7],
    [0.7, 0.4]
], dtype=torch.float32)

labels = torch.tensor(
    [0, 1, 2, 1],
    dtype=torch.long
)


# =====================================
# 3. 定义模型
# =====================================

class FruitClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


# =====================================
# 4. 初始化模型
# =====================================

model = FruitClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# =====================================
# 5. 保存更新前的参数
# =====================================

old_params = {}

for name, param in model.named_parameters():
    old_params[name] = param.detach().clone()


# =====================================
# 6. 清零旧梯度
# =====================================

optimizer.zero_grad()


# =====================================
# 7. Forward
# =====================================

logits = model(x)

predictions = torch.argmax(
    logits,
    dim=1
)

print("Logits：")
print(logits)

print("\n预测类别：")
print(predictions)


# =====================================
# 8. 计算 Loss
# =====================================

loss = criterion(logits, labels)

print("\n更新前 Loss：")
print(loss.item())


# =====================================
# 9. Backward
# =====================================

loss.backward()

print("\n各层参数的梯度：")

for name, param in model.named_parameters():
    print(f"\n参数名称：{name}")
    print("梯度形状：", param.grad.shape)
    print("梯度范数：", param.grad.norm().item())


# =====================================
# 10. Optimizer Step
# =====================================

optimizer.step()


# =====================================
# 11. 比较参数更新前后变化
# =====================================

print("\n参数更新情况：")

for name, param in model.named_parameters():

    old_value = old_params[name]
    new_value = param.detach()

    difference = new_value - old_value

    print(f"\n参数名称：{name}")
    print("参数变化范数：", difference.norm().item())


# =====================================
# 12. 重新计算更新后的 Loss
# =====================================

with torch.no_grad():

    new_logits = model(x)

    new_loss = criterion(
        new_logits,
        labels
    )

print("\n更新后的 Loss：")
print(new_loss.item())


# =====================================
# 13. 最终比较
# =====================================

print("\n========== 最终结果 ==========")

print("更新前 Loss：", loss.item())
print("更新后 Loss：", new_loss.item())

print("本次参数更新已完成")
