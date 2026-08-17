import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss

def validate(model: CNN, val_loader: DataLoader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, paths in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for img_name, pred in zip(paths, preds):
                results.append({'id': img_name, 'prediction': pred})

    df = pd.DataFrame(results)
    df.to_csv('CNN.csv', index=False)
    print(f"Predictions saved to 'CNN.csv'")
    return