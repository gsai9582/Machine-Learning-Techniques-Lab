# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Create dataset
num_samples = 5000
num_classes = 10

X = np.random.rand(num_samples, 784).astype(np.float32)
y = np.random.randint(0, num_classes, num_samples)

# ✅ FIX HERE
X = torch.tensor(X)
y = torch.tensor(y, dtype=torch.long)

# 2. Model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ANN()

# 3. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training
epochs = 10
batch_size = 64
train_acc = []

for epoch in range(epochs):
    correct = 0
    total = 0

    for i in range(0, num_samples, batch_size):
        inputs = X[i:i+batch_size]
        labels = y[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    train_acc.append(accuracy)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")

# 5. Plot
plt.plot(train_acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
