
from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import dim
from vocab import Vocab
from torch import nn, Tensor
import torch
import numpy

# ============ 数据 ============
# 正面 (label=1)
positive = [
    "i love this movie",
    "this is great",
    "amazing work",
    "i am so happy",
    "wonderful day",
    "best thing ever",
    "i really like it",
    "so good",
]

# 负面 (label=0)
negative = [
    "i hate this movie",
    "this is bad",
    "terrible work",
    "i am so sad",
    "awful day",
    "worst thing ever",
    "i really dislike it",
    "so bad",
]

texts = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def buildvocab() -> Vocab :

    all_tokens = set()
    for text in texts:
        all_tokens.update(tokenize(text))

    # 构建 token_to_id 字典
    token_to_id = {
        "<unk>": 0,
        "<pad>": 1,
    }
    for i, token in enumerate(sorted(all_tokens)):
        token_to_id[token] = i + 2

    # 用你的 Vocab 类
    vocab = Vocab(
        token_to_id=token_to_id,
        unk_token="<unk>",
        pad_token="<pad>",
    )

    print(vocab.Summary())
    print(f"词表大小: {len(vocab)}")

    return vocab

class TextClassifier(nn.Module) :
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx) :
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx = pad_idx)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, X) :
        mask = (X != self.embed.padding_idx)
        mask = mask.unsqueeze(-1)
        X = self.embed(X)
        X = (X * mask).sum(dim = 1) / mask.sum(dim = 1)
        X = self.classifier(X)
        return X

def predict(model, vocab, texts_to_test):
    """对新句子进行预测"""
    model.eval()  # 切换到评估模式
    
    # 预处理：tokenize + encode
    encoded = [vocab.encode(tokenize(text)) for text in texts_to_test]
    max_len = max(len(seq) for seq in encoded)
    
    # padding
    padded = [
        seq + [vocab.pad_id] * (max_len - len(seq))
        for seq in encoded
    ]
    
    input_tensor = torch.tensor(padded, dtype=torch.long)
    
    with torch.no_grad():  # 推理时不需要计算梯度
        logits = model(input_tensor)
        predictions = torch.argmax(logits, dim=1)  # 取概率最大的类别
    
    # 输出结果
    label_names = ["负面 👎", "正面 👍"]
    for text, pred in zip(texts_to_test, predictions):
        print(f"'{text}' → {label_names[pred.item()]}")

def main() :
    vocab = buildvocab()
    encode_texts = [vocab.encode(tokenize(text)) for text in texts]
    max_len = max(len(text) for text in encode_texts)
    global labels
    labels = torch.as_tensor(labels, dtype = torch.long)
    padded = [
        seq + [vocab.pad_id] * (max_len - len(seq))
        for seq in encode_texts
    ]

    encode_texts = torch.as_tensor(padded, dtype = torch.long)
    model = TextClassifier(len(vocab), 32, 2, vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 100
    for epoch in range(num_epochs) :
        predictions = model(encode_texts)
        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    print("\n===== 测试 =====")
    test_texts = [
        "i love it",
        "this is terrible", 
        "amazing day",
        "i hate everything",
        "so wonderful",
        "really bad movie",
    ]
    predict(model, vocab, test_texts)

if __name__ == "__main__" :
    main()