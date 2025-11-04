import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter
from torch.optim.lr_scheduler import CosineAnnealingLR

from fed_learning_cifar_experiment.models.resnet_cnn_model import tiny_resnet18
from fed_learning_cifar_experiment.task import test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ColorJitter(0.2, 0.2, 0.2, 0.1),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root="./data", train=True, download=False, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=False, transform=test_transform)

    num_workers = 0 if torch.get_num_threads() == 1 else 2
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = tiny_resnet18(num_classes=10, base_width=16).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    epochs = 120
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        test_loss, test_acc = test(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), "pretrained_cifar.pth")
    final_loss, final_acc = test(model, test_loader, device)
    print(f"\nFinal Centralized Test Accuracy: {final_acc*100:.2f}%")

if __name__ == "__main__":
    main()

