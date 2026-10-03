import torch
from torchvision import transforms
from PIL import Image
import sys

from model import SkinCancerCNN

# Load model
model = SkinCancerCNN()
model.load_state_dict(torch.load("models/skin_cancer_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image path (change this to your test image)
image_path = "C:/Users/supri/Desktop/skin_cancer_classifier/data/images/C0063608-Malignant_tumor_of_the_skin.jpg"
# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    prob = torch.sigmoid(output).item()
    prediction = "Malignant" if prob > 0.5 else "Benign"

print(f"🧠 Prediction: {prediction} (Confidence: {prob:.4f})")