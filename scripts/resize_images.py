import os
from PIL import Image
from tqdm import tqdm

input_dir = "data/images"
output_dir = "data/resized_images"
os.makedirs(output_dir, exist_ok=True)

# Collect all image paths first
image_paths = []
for root, _, files in os.walk(input_dir):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, fname)
            image_paths.append(full_path)

# Resize with progress bar
for full_path in tqdm(image_paths, desc="🔄 Resizing images"):
    rel_path = os.path.relpath(full_path, input_dir)
    save_path = os.path.join(output_dir, rel_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        img = Image.open(full_path).convert("RGB")
        img = img.resize((224, 224))
        img.save(save_path)
    except Exception as e:
        print(f"⚠️ Failed to process {full_path}: {e}")

print("✅ All images resized to 224×224 and saved to data/resized_images/")