import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import create_data_loaders, CustomImageFolder
from model import EfficientNetV2

def train_model(data_dir, num_epochs=5, batch_size=8, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = create_data_loaders(data_dir, batch_size)

    num_classes = len(train_loader.dataset.classes)
    model = EfficientNetV2(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels, paths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs-1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for val_inputs, val_labels, val_paths in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_inputs.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == val_labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs-1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    torch.save(model.state_dict(), 'efficientnet_v2.pth')
    print('Model saved.')

if __name__ == '__main__':
    data_dir = 'data'
    train_model(data_dir)