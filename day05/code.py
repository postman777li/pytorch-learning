import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
# 1. 准备原始数据
texts = [
    "I love AI",
    "This is terrible",
    "AI is amazing",
    "I hate this",
    "Very good work",
    "This is wonderful",
]

labels = [1, 0, 1, 0, 1, 1]

# 2. 自定义 Dataset
class TextClassificationDataset(Dataset):
    def __init__(self,texts,labels):
        if len(texts) != len(labels):
            raise ValueError("文本和标签数量必须一致")
        self.texts=texts
        self.labels=labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,index):
        text=self.texts[index]
        label=self.labels[index]
        return {
            "text":text,
            "label":label
        }

# 3. 创建 Dataset 对象
dataset = TextClassificationDataset(texts, labels)


# 4. 测试 Dataset
print("数据集大小：", len(dataset))

print("第一条数据：", dataset[0])

print("第三条数据：", dataset[2])

train_dataset, val_dataset = random_split(
    dataset,
    [4, 2],
    generator=torch.Generator().manual_seed(42)
)

print("\n训练集大小:", len(train_dataset))
print("验证集大小:", len(val_dataset))