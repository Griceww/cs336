from torch import nn
import torch

class TinyMLP(nn.Module) :
    def __init__(self) :
        super().__init__()
        print(nn.Embedding.__bases__)
        exit
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

def main() :
    print("#")
    X = torch.randn(100, 2)  # 100 个样本，每个样本 2 维
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * torch.randn(100)
    y = y.unsqueeze(-1)
    model = TinyMLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        predictions = model(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__" :
    main()