import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import SkinCancerCNN

# Load model
model = SkinCancerCNN()
model.load_state_dict(torch.load("models/skin_cancer_model.pt", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Streamlit UI
st.title("🧠 Skin Cancer Classifier")
st.write("Upload a skin lesion image to predict whether it's **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Malignant" if prob > 0.5 else "Benign"

    st.markdown(f"### 🩺 Prediction: **{prediction}**")
    st.markdown(f"Confidence: `{prob:.4f}`")