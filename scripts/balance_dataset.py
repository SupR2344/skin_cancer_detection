import pandas as pd

df = pd.read_csv("data/filtered_labels.csv")

# Separate classes
benign = df[df["target"] == 0]
malignant = df[df["target"] == 1]

# Undersample benign
benign_sampled = benign.sample(n=len(malignant), random_state=42)

# Combine and shuffle
balanced_df = pd.concat([benign_sampled, malignant]).sample(frac=1, random_state=42)

# Save new CSV
balanced_df.to_csv("data/balanced_labels.csv", index=False)

print(f"✅ Saved balanced dataset with {len(balanced_df)} samples")