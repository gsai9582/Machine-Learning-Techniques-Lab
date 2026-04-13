# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# 4. Model function
class MLP(nn.Module):
    def __init__(self, input_size, hidden_units):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 5. Training function
def train_model(hidden_units=16, lr=0.001, epochs=10):
    model = MLP(30, hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Accuracy
    with torch.no_grad():
        preds = model(X_test)
        predicted = (preds > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / len(y_test)

    return accuracy

# 6. Baseline
baseline_acc = train_model()
print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")

# 7. Hyperparameter Grid
param_grid = {
    'hidden_units': [8, 16, 32],
    'lr': [0.01, 0.001],
    'epochs': [10, 20]
}

# 8. Grid Search
best_acc = 0
best_params = None

for params in ParameterGrid(param_grid):
    acc = train_model(
        hidden_units=params['hidden_units'],
        lr=params['lr'],
        epochs=params['epochs']
    )
    
    if acc > best_acc:
        best_acc = acc
        best_params = params

print("\nBest Parameters:", best_params)
print(f"Optimized Accuracy: {best_acc*100:.2f}%")
