import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import models

# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================================================
# MULTIMODAL DATASET
# ============================================================

class MultiModalCIFAR(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        # ====================================================
        # METADATA = AVERAGE BRIGHTNESS
        # ====================================================

        brightness = image.mean().unsqueeze(0)

        return image, brightness, label

# ============================================================
# LOAD CIFAR10 DATASET
# ============================================================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# ============================================================
# REDUCE DATASET SIZE
# ============================================================

train_dataset = Subset(train_dataset, range(1000))
test_dataset = Subset(test_dataset, range(200))

# ============================================================
# CONVERT TO MULTIMODAL DATASET
# ============================================================

train_dataset = MultiModalCIFAR(train_dataset)
test_dataset = MultiModalCIFAR(test_dataset)

# ============================================================
# SHOW SAMPLE METADATA
# ============================================================

sample_image, sample_metadata, sample_label = train_dataset[0]

print("\n===================================")
print(" SAMPLE METADATA ")
print("===================================\n")

print(f"Brightness Value : {sample_metadata.item():.4f}")
print(f"Class Label      : {sample_label}")

# ============================================================
# SAVE METADATA RESULT
# ============================================================

with open("output/metadata_result.txt", "w") as f:

    f.write("MULTIMODAL METADATA RESULT\n")
    f.write("===========================\n\n")

    f.write(
        f"Brightness Value : "
        f"{sample_metadata.item():.4f}\n"
    )

    f.write(
        f"Class Label      : "
        f"{sample_label}\n"
    )

print("\nMetadata saved to output/metadata_result.txt")

# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# MULTIMODAL RESNET18
# ============================================================

class MultiModalResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.resnet = models.resnet18(weights=None)

        # Remove original FC
        self.resnet.fc = nn.Identity()

        # Fusion layer
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, image, metadata):

        features = self.resnet(image)

        combined = torch.cat(
            [features, metadata],
            dim=1
        )

        output = self.classifier(combined)

        return output

# ============================================================
# MULTIMODAL VISION TRANSFORMER
# ============================================================

class MultiModalViT(nn.Module):

    def __init__(self):

        super().__init__()

        self.vit = models.vit_b_16(weights=None)

        # Remove original head
        self.vit.heads = nn.Identity()

        # Fusion layer
        self.classifier = nn.Sequential(
            nn.Linear(768 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, image, metadata):

        features = self.vit(image)

        combined = torch.cat(
            [features, metadata],
            dim=1
        )

        output = self.classifier(combined)

        return output

# ============================================================
# CREATE MODELS
# ============================================================

cnn_model = MultiModalResNet()

vit_model = MultiModalViT()

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(model, loader, epochs=EPOCHS):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    model.to(DEVICE)

    start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, metadata, labels in loader:

            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images, metadata)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        acc = correct / total

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {running_loss / len(loader):.4f} | "
            f"Accuracy: {acc:.4f}"
        )

    elapsed = time.time() - start

    return elapsed

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, metadata, labels in loader:

            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, metadata)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    return correct / total

# ============================================================
# PARAMETER COUNTER
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print(" MULTIMODAL LAB 3.4 ")
    print(" RESNET18 vs VISION TRANSFORMER ")
    print("===================================\n")

    # ========================================================
    # TRAIN RESNET18
    # ========================================================

    print("\nTraining ResNet18...\n")

    print(
        f"ResNet18 Parameters: "
        f"{count_parameters(cnn_model):,}"
    )

    cnn_time = train_model(
        cnn_model,
        train_loader
    )

    cnn_acc = evaluate_model(
        cnn_model,
        test_loader
    )

    print(f"\nResNet18 Test Accuracy: {cnn_acc:.4f}")

    # ========================================================
    # TRAIN VISION TRANSFORMER
    # ========================================================

    print("\nTraining Vision Transformer...\n")

    print(
        f"Vision Transformer Parameters: "
        f"{count_parameters(vit_model):,}"
    )

    vit_time = train_model(
        vit_model,
        train_loader
    )

    vit_acc = evaluate_model(
        vit_model,
        test_loader
    )

    print(f"\nVision Transformer Test Accuracy: {vit_acc:.4f}")

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n===================================")
    print(" FINAL COMPARISON ")
    print("===================================\n")

    print(f"ResNet18 Accuracy      : {cnn_acc:.4f}")
    print(f"Vision Transformer Acc : {vit_acc:.4f}")

    print(
        f"\nResNet18 Parameters      : "
        f"{count_parameters(cnn_model):,}"
    )

    print(
        f"Vision Transformer Params: "
        f"{count_parameters(vit_model):,}"
    )

    print(
        f"\nResNet18 Training Time      : "
        f"{cnn_time:.1f} s"
    )

    print(
        f"Vision Transformer Time     : "
        f"{vit_time:.1f} s"
    )

    # ========================================================
    # BAR CHART
    # ========================================================

    model_names = [
        "ResNet18",
        "Vision Transformer"
    ]

    accuracies = [
        cnn_acc,
        vit_acc
    ]

    plt.figure(figsize=(6, 4))

    plt.bar(model_names, accuracies)

    plt.ylabel("Accuracy")

    plt.title("Multimodal Model Accuracy")

    plt.savefig("output/vit_vs_resnet.png")

    print("\nGraph saved to output/vit_vs_resnet.png")

    plt.show()