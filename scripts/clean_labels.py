import pandas as pd

df = pd.read_csv("data/balanced_labels.csv")

# Drop rows with missing image names or literal "nan"
df = df.dropna(subset=["image_name"])
df = df[~df["image_name"].isin(["nan", "nan.jpg", "NaN", ""])]
df = df[df["image_name"].str.endswith(".jpg")]

# Save cleaned version
df.to_csv("data/balanced_labels.csv", index=False)
print(f"✅ Deep cleaned labels: {len(df)} entries saved")