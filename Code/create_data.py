import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from glob import glob

DATA_PATH = "gestures"

def load_images():
    images = []
    labels = []
    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, 0)  # grayscale
            if img is None:
                continue
            img = cv2.resize(img, (50, 50))  # resize if needed
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

print("[INFO] Loading images...")
images, labels = load_images()
print(f"[INFO] Loaded {len(images)} images.")

# Split into training and validation
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42)

# Save datasets
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)

print("[SUCCESS] Dataset pickles saved!")
