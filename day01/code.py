import torch

#创建tensor

#0维
a=torch.tensor(5) 
#1维  这不就是一维数组吗
b=torch.tensor([1,2])
#2维
c=torch.tensor([
    [1,2],
    [2,3]
])
print(a)
print(b)
print(c)

#维度
print(c.ndim)
#形状
print(c.shape)

# 全部为 0
d=torch.zeros(2,3)
# 全部为 1
d=torch.ones(2,3)
# 全部为指定数字
d=torch.full((2,3),7)
# 随机生成 0~1 之间的数字
d=torch.rand(2,3)
# 随机生成符合标准正态分布的数字
d=torch.randn(2,3)
# 生成连续整数
d=torch.arange(0,10)
# 生成单位矩阵
d=torch.eye(5)

#数据类型
print(d.dtype)
print(d.device)
print(d.numel())

scores = torch.tensor([
    [85, 90, 78],
    [92, 88, 95],
    [76, 80, 85]
])
# 取第一行
print(scores[0])
# 取第二行第三列
print(scores[1,2])


#切片 Slicing  tensor[start:end] 
# 表示从 start 开始，取到 end 之前，不包含 end。

# 取前两行
print(scores[0:2])
# 取所有行的第一列
print(scores[:,0])
# 取所有行的前两列
print(scores[:,0:2])
# 取前两行的前两列
print(scores[0:2,0:2])

x = torch.tensor([1, 2, 3, 4])
x[0]=100
x[0:2]=1000

#广播机制 Broadcasting
# PyTorch 的广播规则：
# 从最后一个维度开始，逐维比较两个 Tensor 的 Shape。

# 每个维度必须满足以下条件之一：
# 1. 两个维度大小相等。
# 2. 其中一个维度大小为 1。
# 3. 其中一个 Tensor 没有这个维度，此时可以视为在前面补 1。
scores = torch.tensor([
    [80, 90, 70],
    [85, 75, 95],
    [90, 80, 85]
])
bonus = torch.tensor([5, 10, 3])

result=scores+bonus


#Tensor 的数学运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a - b)
#逐元素相乘
print(a * b)
print(a / b)

a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = torch.tensor([
    [5, 6],
    [7, 8]
])

#矩阵乘法
c = a @ b
c=torch.matmul(a,b)
print(c)

#数据类型转换
a.int()
a.float()
a.double()
a.bool()