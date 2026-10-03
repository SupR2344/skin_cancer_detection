import pandas as pd

benign = pd.read_csv("data/balanced_labels.csv")
malignant = pd.read_csv("data/malignant_labels.csv")

combined = pd.concat([benign, malignant], ignore_index=True)
combined.to_csv("data/balanced_labels.csv", index=False)
print(f"✅ Combined dataset saved: {len(combined)} total images")