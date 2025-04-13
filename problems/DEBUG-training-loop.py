'''Debug this script without running it'''

import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset (100 samples, 10 features, binary labels)
inputs = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))
train_dataset = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)

# Simple model definition
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)  # two output classes (binary classification)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 5
for epoch in range(epochs):
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} completed")

# Evaluate on training set
correct = 0
total = 0
with torch.no_grad():
    for data, labels in train_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = 100 * correct / total

# Compare with baseline
print(f"Accuracy improvement over baseline: {accuracy - baseline_acc:.2f}%")
