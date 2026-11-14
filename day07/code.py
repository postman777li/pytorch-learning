import torch
# ==========================================
# 1. 准备原始文本
# ==========================================

texts = [
    "I love AI",
    "I love deep learning",
    "Hello"
]


# ==========================================
# 2. 构建词表 Vocabulary
# ==========================================

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "I": 2,
    "love": 3,
    "AI": 4,
    "deep": 5,
    "learning": 6,
    "Hello": 7
}

PAD_ID = vocab["[PAD]"]
UNK_ID = vocab["[UNK]"]

# ==========================================
# 3. 定义简单 Tokenizer
# ==========================================

class SimpleTokenizer:
    def __init__(self,vocab):
        self.vocab=vocab
        self.pad_id = vocab["[PAD]"]
        self.unk_id = vocab["[UNK]"]

    def tokenize(self,text):
        # 按空白字符切分文本
        texts=text.split()
        return texts

    def convert_tokens_to_ids(self, tokens):
        ids=[]
        for token in tokens:
            token_id=self.vocab.get(token,self.unk_id)
            ids.append(token_id)

        return ids

    def encode(self,text,max_length=5):
        # Step 1: 文本切分
        tokens=self.tokenize(text)
        
        # Step 2: 转换为 Token ID
        input_ids=self.convert_tokens_to_ids(tokens)

        # Step 3: Truncation
        input_ids=input_ids[:max_length]

        # Step 4: 构造 Attention Mask
        attention_mask=[1]*len(input_ids)

        # Step 5: 计算 Padding 长度
        padding_length=max_length-len(input_ids)

        # Step 6: Padding
        input_ids=(input_ids+[self.pad_id]*padding_length)

        attention_mask = (
            attention_mask
            + [0] * padding_length
        )

        # Step 7: 返回编码结果
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# ==========================================
# 4. 创建 Tokenizer
# ==========================================
tokenizer=SimpleTokenizer(vocab)

# ==========================================
# 5. 单条文本编码
# ==========================================

text = "I love AI"

tokens=tokenizer.tokenize(text)
print("原始文本:", text)
print("Tokens:", tokens)

encoded=tokenizer.encode(text,max_length=5)

print("Input IDs:", encoded["input_ids"])
print("Attention Mask:", encoded["attention_mask"])


# ==========================================
# 6. 批量编码文本
# ==========================================

all_input_ids = []
all_attention_masks = []

for text in texts:

    encoded = tokenizer.encode(
        text,
        max_length=5
    )

    all_input_ids.append(
        encoded["input_ids"]
    )

    all_attention_masks.append(
        encoded["attention_mask"]
    )

# ==========================================
# 7. 转换为 PyTorch Tensor
# ==========================================

input_ids_tensor = torch.tensor(
    all_input_ids,
    dtype=torch.long
)

attention_mask_tensor = torch.tensor(
    all_attention_masks,
    dtype=torch.long
)

# ==========================================
# 8. 查看最终结果
# ==========================================

print("\n===== Batch 输入 =====")

print("input_ids:")
print(input_ids_tensor)

print("\nattention_mask:")
print(attention_mask_tensor)

print("\ninput_ids shape:")
print(input_ids_tensor.shape)

print("\nattention_mask shape:")
print(attention_mask_tensor.shape)

# ==========================================
# 9. 模拟 Embedding 层
# ==========================================

embedding = torch.nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=8,
    padding_idx=PAD_ID
)

embedded = embedding(input_ids_tensor)

print("\nEmbedding 输出 Shape:")
print(embedded.shape)