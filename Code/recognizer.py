import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and label map
model = load_model("cnn_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the region of interest (ROI) for gesture detection
    roi = gray[100:300, 100:300]  # You can adjust this part based on your requirements

    # Resize ROI to 64x64 (model input size)
    roi_resized = cv2.resize(roi, (64, 64))

    # Normalize and reshape for prediction
    roi_resized = roi_resized.astype("float32") / 255.0
    roi_resized = roi_resized.reshape(1, 64, 64, 1)

    # Predict the gesture only once per frame
    predictions = model.predict(roi_resized)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]

    # Get the label for the predicted class
    label = label_map[class_id]

    # Display the result
    cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
