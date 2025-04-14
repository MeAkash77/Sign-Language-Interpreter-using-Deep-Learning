import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera 0 failed. Trying next one...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Still couldn't open camera. Try updating drivers or plugging in webcam.")
else:
    print("✅ Camera opened! Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Test Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
