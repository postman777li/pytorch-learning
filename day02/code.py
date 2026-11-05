import torch

#-----------------------------------------------------------------
x=torch.tensor(2.0)
y=torch.tensor(10.0)
#是否需要追踪梯度
w=torch.tensor(3.0,requires_grad=True)

y_pred=x*w

loss=(y_pred-y)**2
#触发反向传播 从 Loss 开始，沿着计算关系反向计算梯度
loss.backward()
print("预测值:", y_pred.item())
print("Loss:", loss.item())
#计算得到的梯度 将单元素 Tensor 转为 Python 数值
print("梯度:", w.grad.item())

#-----------------------------------------------------------------
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b
d = c + 4
loss = d ** 2

loss.backward()

print(a.grad)  # tensor(60.)
print(b.grad)  # tensor(40.)


#-----------------------------------------------------------------
w = torch.tensor(2.0, requires_grad=True)

loss1 = w ** 2
loss1.backward()
print(w.grad)  # 4

w.grad = None

loss2 = w ** 2
loss2.backward()
print(w.grad)  # 4

# backward()：计算并累积梯度。
# zero_grad()：清理已有梯度。
# optimizer.step()：根据已有梯度更新参数。

#-----------------------------------------------------------------
w = torch.tensor(3.0, requires_grad=True)

y = w * 2
#detach() 不是让 y 消失，也不是清空 y 的数值，而是得到一个新的 Tensor，
# 它拥有相同的数据，但不再携带 y 原来的求导关系。 就是没梯度了
z = y.detach()

print(y)
print(z)

print(y.requires_grad)
print(z.requires_grad)

#-----------------------------------------------------------------
w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
#这一段代码，不要记录用于自动求导的计算过程
with torch.no_grad():
    y = w * x

print(y)
print(y.requires_grad)
#-----------------------------------------------------------------
w = torch.tensor(3.0, requires_grad=True)

with torch.no_grad():
    a = w * 2

b = w * 3

print(a.requires_grad)  # False
print(b.requires_grad)  # True

#-----------------------------------------------------------------
#完整训练代码
x=torch.tensor(2.0)
y=torch.tensor(10.0)

w=torch.tensor(3.0,requires_grad=True)
lr=0.1

# 4. 开始训练
for epoch in range(10):
    y_pred=w*x
    loss=(y_pred-y)**2
    #清空梯度
    w.grad=None
    # Backward: 反向计算梯度
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
    print(
        f"Epoch {epoch + 1}: "
        f"w={w.item():.4f}, "
        f"loss={loss.item():.4f}"
    )