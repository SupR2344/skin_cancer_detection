import os
import cv2
import pandas as pd
from tqdm import tqdm

def resize_and_save_images(input_dir="data/images/train", output_dir="data/resized_images", size=224):
    df = pd.read_csv("data/processed_labels.csv")
    os.makedirs(output_dir, exist_ok=True)

    for image_id in tqdm(df["image_id"]):
        img_path = os.path.join(input_dir, f"{image_id}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (size, size))
            out_path = os.path.join(output_dir, f"{image_id}.jpg")
            cv2.imwrite(out_path, img)
        else:
            print(f"⚠️ Missing: {img_path}")