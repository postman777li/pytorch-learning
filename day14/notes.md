# Day 14 · Adam 与 AdamW

## 核心机制

Adam 用梯度历史确定方向，并按各参数的梯度尺度调整更新幅度。

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

| 机制 | 作用 | 常见参数 |
|---|---|---|
| 一阶矩 m | 平滑梯度方向 | β1=0.9 |
| 二阶矩 v | 估计梯度平方的尺度 | β2=0.999 |
| 偏差修正 | 补偿从0初始化的早期偏差 | 使用当前更新步数 t |
| ε | 提高除法的数值稳定性 | 1e-8 |

例子中第一步 g=2，m=0.2，经偏差修正得到 m̂=2。

## 权重衰减的区别

| 优化器 | 衰减处理 | 是否进入一阶、二阶矩 |
|---|---|---|
| 传统 Adam，耦合 L2 | 用 g+λθ 参与更新 | 是 |
| AdamW | 对参数独立施加 -ηλθ | 否 |

仅考虑衰减，θ=10、lr=0.1、λ=0.01，参数收缩到9.99。AdamW 便于独立控制正则化和自适应更新，大模型中很常见。Bias、LayerNorm 参数常使用不衰减的参数组。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

训练循环继续使用 zero_grad → backward → step。Adam 系列需要保存 m、v，优化器状态增加内存开销。较新 PyTorch 的 Adam 也提供解耦衰减选项，具体以所用 API 配置为准。

## 本日代码

[code.py](code.py) 直接取自《学习SGD优化器》的 Day14 完整示例。三组实验使用同一初始权重、同一打乱顺序，对比 SGD、Adam、AdamW。其学习率与衰减配置有差异，结果用于观察这些配置的表现，结论范围限于本次任务和超参数。
