# Day 16 · Learning Rate Scheduler

Scheduler 按训练进度修改优化器的学习率，优化器使用当前 lr 更新参数。

| 策略 | 变化规律 | 典型用途 |
|---|---|---|
| Warmup | 初期从小逐渐增大 | 缓和训练初期更新 |
| Linear Decay | 随进度线性减小 | 常见微调调度 |
| Cosine Decay | 开始和结束平缓，中段下降较快 | 预训练及其他训练 |
| StepLR | 每 step_size 次调用乘 gamma | 按 Epoch 或指定频率阶梯衰减 |

衰减进度 p 从0到1，余弦调度可表示为下式。

$$
lr(p)=lr_{min}+\frac{lr_{max}-lr_{min}}2(1+\cos(\pi p))
$$

## 工程中优先使用封装

原窗口补充问答推荐 Hugging Face 的成熟接口。

```python
from transformers import get_scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)
```

也可用 get_cosine_schedule_with_warmup。上述默认 cosine 衰减至0，带非零最低 lr 的方案需选择支持该配置的接口。调用封装前安装 transformers。

## 调用顺序与步数

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

先用当前 lr 更新参数，再设置下次更新的 lr。按 Epoch 的 StepLR 放在 Epoch 结束处，按更新步的 Warmup/Cosine 每次参数更新后推进。梯度累积时，按真实 optimizer 更新次数计数，尾批及不足一个累积周期也需计入。

无梯度累积时，total_steps=num_epochs*len(loader)。LambdaLR 返回基础 lr 的倍率。创建时就会初始化学习率，检查第一步使用的值，日志可在 optimizer.step 前记录 optimizer.param_groups[0]["lr"]。

## 本日代码

[code.py](code.py) 直接取自 Day16 的完整 lr_scheduler.py，用 LambdaLR 演示 Warmup + Cosine，峰值 lr=0.01，前10%步数预热，最低比例0.1，共30轮。它保留窗口的手写教学版本，工程选型可直接采用上面的封装接口。

运行需要 PyTorch 与 Matplotlib，程序显示学习率与 Loss 曲线。窗口代码是教学示例，本模块记录为已学习、教学代码归档。
