import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from CNN import CNN, train, validate  # 假設你之前train、validate已經寫好
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset  # 你原本的
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

# ====== Settings ======
pretrained_model_path = 'best_cnn_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fine_tune_all_layers = True  # <-- 這裡改 True / False 決定是否要全部fine-tune

batch_size = 32
learning_rate = 1e-5  # 微調時學習率設小一點
EPOCHS = 10  # 再微調10個epochs（可以自己調）

# ====== Load dataset ======
train_images, train_labels = load_train_dataset()
train_images, train_labels = shuffle(train_images, train_labels, random_state=777)

train_len = int(0.8 * len(train_images))
train_dataset = TrainDataset(train_images[:train_len], train_labels[:train_len])
val_dataset = TrainDataset(train_images[train_len:], train_labels[train_len:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ====== Build model and load pretrained weights ======
model = CNN(num_classes=5).to(device)
model.load_state_dict(torch.load(pretrained_model_path))
print("✅ Successfully loaded pretrained model.")

# ====== Decide fine-tune strategy ======
if not fine_tune_all_layers:
    # 凍結除了fc層以外的全部參數
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    print("🔒 Only fine-tuning the FC layers.")

# ====== Optimizer只訓練可訓練參數 ======
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ====== Fine-tuning ======
best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'fine_tuned_cnn_model.pth')  # 存最好的model
        print("💾 Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

print("🎯 Fine-tuning complete.")