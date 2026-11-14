import torch
from torch.utils.data import Dataset,DataLoader

# ==========================================
# 1. 准备 Tokenizer 编码后的数据
# ==========================================

input_ids_list = [
    [2, 3, 4],
    [2, 3, 5, 6, 7],
    [8, 9],
    [2, 4, 5, 6],
    [8, 3]
]

labels = [1, 0, 1, 0, 1]

# ==========================================
# 2. 定义 Dataset
# ==========================================

class TextDataset(Dataset):

    def __init__(self, input_ids_list, labels):

        if len(input_ids_list) != len(labels):
            raise ValueError("数据和标签数量不一致")

        self.input_ids_list = input_ids_list
        self.labels = labels

    def __len__(self):

        return len(self.input_ids_list)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids_list[index],
            "label": self.labels[index]
        }



# ==========================================
# 3. 定义 Data Collator
# ==========================================
class MyDataCollator:
    def __init__(self,pad_token_id=0):
        self.pad_token_id=pad_token_id

    def __call__(self,samples):
         # 找出当前 Batch 中最长的序列长度
        max_length=max(
            len(sample["input_ids"])
            for sample in samples
        )

        # 创建空列表，保存整理后的数据
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for sample in samples:
            # 获取当前样本的 Token ID
            input_ids=sample["input_ids"]

            # 计算需要填充多少个 PAD
            padding_length=max_length-len(input_ids)

            # 真实 Token 对应 1
            attention_mask=[1]*len(input_ids)

            # Padding：在右侧补充 PAD
            input_ids = (
                input_ids
                + [self.pad_token_id] * padding_length
            )

            # Mask：在右侧补充 0
            attention_mask = (
                attention_mask
                + [0] * padding_length
            )

            # 保存整理后的数据
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

            # 保存标签
            batch_labels.append(sample["label"])

        # 将列表转换为 Tensor
        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long
            )
        }

# ==========================================
# 4. 创建 Dataset
# ==========================================

dataset = TextDataset(
    input_ids_list,
    labels
)


# ==========================================
# 5. 创建 Data Collator
# ==========================================

collator = MyDataCollator(
    pad_token_id=0
)


# ==========================================
# 6. 创建 DataLoader
# ==========================================

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=collator
)


# ==========================================
# 7. 遍历 Batch
# ==========================================

for batch_index, batch in enumerate(loader):

    print("\n========== Batch", batch_index, "==========")

    print("\nInput IDs:")
    print(batch["input_ids"])

    print("\nAttention Mask:")
    print(batch["attention_mask"])

    print("\nLabels:")
    print(batch["labels"])

    print("\n各部分 Shape:")
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
