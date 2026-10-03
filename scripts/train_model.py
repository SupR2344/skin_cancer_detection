import torch
from scripts.model import SkinCancerCNN
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from image_loader import load_images
from model import build_model

model = SkinCancerCNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

df = pd.read_csv("data/processed_labels.csv")
X, y = load_images(df, "data/images")
y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

model = build_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
model.save("models/skin_cancer_model.h5")