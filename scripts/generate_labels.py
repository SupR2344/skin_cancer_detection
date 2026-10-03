import os
import pandas as pd

image_dir = "data/images"
data = []

for label_folder in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, label_folder)
    if not os.path.isdir(folder_path):
        continue

    folder_lower = label_folder.lower()
    if "malignant" in folder_lower:
        label = 1
    elif "benign" in folder_lower:
        label = 0
    else:
        continue  # Skip unknown folders

    # Recursively walk through subfolders
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, fname), image_dir)
                data.append({
                    "image_name": rel_path.replace("\\", "/"),  # Normalize path
                    "target": label
                })

df = pd.DataFrame(data)
df.to_csv("data/balanced_labels.csv", index=False)
print(f"✅ Created CSV with {len(df)} entries — {sum(df['target']==1)} malignant, {sum(df['target']==0)} benign")