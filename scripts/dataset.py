import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SkinCancerDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)

        # ✅ Use image_name instead of image_id
        self.df = self.df.dropna(subset=["image_name"])
        self.df = self.df[~self.df["image_name"].isin(["nan", "NaN", ""])]
        self.df = self.df[self.df["image_name"].apply(lambda x: isinstance(x, str) and x.strip().endswith(".jpg"))]

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["image_name"]
        label = self.df.iloc[idx]["target"]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label