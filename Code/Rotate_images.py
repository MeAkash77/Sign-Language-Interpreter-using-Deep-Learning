import cv2
import os

def flip_images():
    gestures_path = "gestures"
    
    for folder in os.listdir(gestures_path):
        folder_path = os.path.join(gestures_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            print(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"⚠️  Warning: Could not read {img_path}, skipping.")
                continue

            flipped = cv2.flip(img, 1)  # flip horizontally
            name, ext = os.path.splitext(img_file)
            new_path = os.path.join(folder_path, f"{name}_flipped{ext}")
            cv2.imwrite(new_path, flipped)

flip_images()
