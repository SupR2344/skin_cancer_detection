import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the combined ground truth file
df = pd.read_csv("data/ground_truth.csv")

# Rename image_name to image_id for consistency
df.rename(columns={"image_name": "image_id"}, inplace=True)

# Encode diagnosis labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["diagnosis"])

# Save processed labels
df.to_csv("data/processed_labels.csv", index=False)

print("✅ Preprocessing complete. Saved to data/processed_labels.csv")