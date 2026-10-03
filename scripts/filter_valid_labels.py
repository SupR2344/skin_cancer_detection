import pandas as pd
import os

# Load full label file
df = pd.read_csv("data/processed_labels.csv")

# Keep only entries with resized images
valid_ids = [
    img_id for img_id in df["image_id"]
    if os.path.exists(f"data/resized_images/{img_id}.jpg")
]

df_filtered = df[df["image_id"].isin(valid_ids)]
df_filtered.to_csv("data/filtered_labels.csv", index=False)

print(f"✅ Saved {len(df_filtered)} valid entries to data/filtered_labels.csv")