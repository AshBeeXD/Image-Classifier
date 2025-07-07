import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train(model, train_loader, val_loader, num_epochs=30, lr=3e-4, device='cpu', save_path='outputs/model.pth'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss/(total//train_loader.batch_size), acc=100.*correct/total)

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"\nValidation Accuracy: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


def evaluate(model, val_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return acc
