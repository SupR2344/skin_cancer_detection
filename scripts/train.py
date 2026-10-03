import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os

from dataset import SkinCancerDataset
from model import SkinCancerCNN

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = SkinCancerDataset(
    csv_path="data/balanced_labels.csv",
    image_dir="data/resized_images",
    transform=transform
)

# Count class distribution
labels = [label for _, label in dataset]
benign_count = labels.count(0)
malignant_count = labels.count(1)
print(f"Benign: {benign_count}, Malignant: {malignant_count}")

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model setup
model = SkinCancerCNN().to(device)

# Loss with pos_weight
pos_weight = torch.tensor([benign_count / malignant_count], device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    train_acc = correct / total
    print(f"✅ Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Accuracy = {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"🔍 Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/skin_cancer_model.pt")
print("📦 Model saved to models/skin_cancer_model.pt")