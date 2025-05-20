import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def train_model(self, df):
        X = np.array(df.select("image").collect()).reshape(-1, 784).astype(np.float32)
        y = np.array(df.select("label").collect()).astype(np.int64)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(5):
            optimizer.zero_grad()
            out = self.net(X_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(self.net(X_tensor), dim=1).numpy()
        acc = accuracy_score(y, preds)
        return preds, acc
