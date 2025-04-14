import cv2
import os
import random
import numpy as np

def get_image_size():
    # Attempt to load a sample image to get the size (this should be from a valid path)
    img = cv2.imread('gestures/0/100.jpg', 0)
    if img is None:
        raise FileNotFoundError("The sample image was not found. Please check the file path.")
    return img.shape

gestures = os.listdir('gestures/')
gestures.sort(key=int)

# Define the range of gestures to display in each row
begin_index = 0
end_index = 5

# Get image dimensions
try:
    image_x, image_y = get_image_size()
except FileNotFoundError as e:
    print(e)
    exit()

# Calculate the number of rows based on the number of gesture directories
rows = len(gestures) // 5 + (1 if len(gestures) % 5 != 0 else 0)

# Initialize the full image to be populated
full_img = None

# Iterate through the rows
for i in range(rows):
    col_img = None
    # Ensure that the index does not exceed the length of the gestures list
    for j in range(begin_index, min(end_index, len(gestures))):
        gesture_dir = gestures[j]  # Get the gesture directory (e.g., '0', '1', ...)
        img_path = f"gestures/{gesture_dir}/{random.randint(1, 1200)}.jpg"  # Randomly pick a file in the directory
        
        # Try to load the image
        img = cv2.imread(img_path, 0)
        
        # If the image is not found, print a message and skip this image
        if img is None:
            print(f"Image not found: {img_path}. Skipping this image.")
            continue  # Skip this image and go to the next one

        # If column image is None, initialize it with the first image, else stack the images horizontally
        if col_img is None:
            col_img = img
        else:
            col_img = np.hstack((col_img, img))

    # Update the index for the next set of gestures in the column
    begin_index += 5
    end_index += 5
    
    # If full image is None, initialize it with the first column, else stack vertically
    if full_img is None:
        full_img = col_img
    else:
        full_img = np.vstack((full_img, col_img))

# Check if full_img is valid (not empty) before displaying
if full_img is None or full_img.size == 0:
    print("No valid images to display. Exiting.")
else:
    # Show the final full image and save it
    cv2.imshow("Gestures", full_img)
    cv2.imwrite('full_img.jpg', full_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
