
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


# ==========================================
# 1. 准备原始数据
# ==========================================

texts = [
    "I love machine learning",
    "This movie is boring",
    "Deep learning is amazing",
    "I dislike this product",
    "Python is very useful",
    "This experience is terrible",
    "AI is the future",
    "I hate waiting"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]


# ==========================================
# 2. 定义 Dataset（复习 Day 5）
# ==========================================

class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):

        if len(texts) != len(labels):
            raise ValueError("文本与标签数量不一致")

        self.texts = texts
        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        return {
            "text": self.texts[index],
            "label": self.labels[index]
        }


# ==========================================
# 3. 创建 Dataset
# ==========================================

dataset = TextClassificationDataset(texts, labels)

print("数据集大小:", len(dataset))


# ==========================================
# 4. 创建 DataLoader
# ==========================================

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)


# ==========================================
# 5. 遍历每个 Batch
# ==========================================

print("\n===== 顺序加载 =====")

for batch_index, batch in enumerate(loader):

    print("\nBatch:", batch_index)

    batch_texts = batch["text"]
    batch_labels = batch["label"]

    print("文本:", batch_texts)
    print("标签:", batch_labels)
    print("Batch大小:", len(batch_texts))


# ==========================================
# 6. 开启 Shuffle
# ==========================================

shuffle_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    num_workers=0
)

print("\n===== 随机加载 =====")

for batch_index, batch in enumerate(shuffle_loader):

    print("\nBatch:", batch_index)
    print("文本:", batch["text"])
    print("标签:", batch["label"])


# ==========================================
# 7. 使用 Sampler
# ==========================================

sampler = RandomSampler(dataset)

sampler_loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler,
    num_workers=0
)

print("\n===== RandomSampler =====")

for batch in sampler_loader:
    print(batch)


# ==========================================
# 8. 测试 drop_last
# ==========================================

drop_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    drop_last=True
)

print("\n===== drop_last=True =====")

for batch_index, batch in enumerate(drop_loader):

    print(
        "Batch:", batch_index,
        "样本数:", len(batch["text"])
    )


# ==========================================
# 9. 查看 Batch 总数
# ==========================================

print("\n数据集样本数:", len(dataset))
print("正常 Batch 数:", len(loader))
print("drop_last=True Batch 数:", len(drop_loader))
