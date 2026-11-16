# Day 13 · SGD 与 Momentum

SGD 根据当前梯度更新参数，学习率控制步长。概念按本模块主题整理，原窗口 Day13 代码待取得后归档。

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

| 内容 | 重点 | 例子 |
|---|---|---|
| 梯度 g | 局部变化方向与速率 | g=-16，减去负数会增大参数 |
| 学习率 η | 控制沿梯度反方向移动多少 | w=3、η=0.1，更新后4.6 |
| 随机梯度下降 | 通常用随机小批估计梯度 | DataLoader(shuffle=True) |
| Momentum | 累积历史梯度趋势，减少来回振荡 | momentum=0.9 |

无 dampening、无权重衰减的常见 Momentum 写法如下。

$$
v_t=\mu v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t
$$

μ 控制历史影响。当前梯度为0时，已有动量仍可能推动参数变化。学习率过大可能越过低损失区域，Momentum 也需要配合适当学习率。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

普通 SGD 不含动量时，参数变化应约等于 -lr*grad。手动更新需放在 no_grad 中，backward 负责计算并累积梯度，step 使用梯度更新参数。
