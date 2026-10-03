import torch
from torchvision import transforms
from PIL import Image
from model import SkinCancerCNN

# ✅ Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load model before defining the function
model = SkinCancerCNN()
model.load_state_dict(torch.load("models/skin_cancer_model.pt", map_location=device))
model.to(device)
model.eval()

# ✅ Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Define prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)  # Apply sigmoid manually
        prediction = prob.item()

    label = "malignant" if prediction > 0.5 else "benign"
    print(f"🩺 Prediction: {label} ({prediction:.4f})")