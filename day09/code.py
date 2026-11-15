import torch
import torch.nn as nn
# 4条样本，每条样本有2个特征
x=torch.tensor([
    [0.8, 0.6],
    [0.9, 0.3],
    [0.5, 0.7],
    [0.7, 0.4]
],dtype=torch.float32)

# 对应的真实标签
labels = torch.tensor(
    [0, 1, 2, 1],
    dtype=torch.long
)

class FruitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self,x):
        return self.network(x)

model=FruitClassifier()

logits=model(x)

print(logits)
print(logits.shape)

predictions=torch.argmax(logits,dim=1)
print(predictions)

criterion=nn.CrossEntropyLoss()
loss=criterion(logits,labels)

print(loss)

