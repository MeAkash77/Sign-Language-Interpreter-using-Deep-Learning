import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            roi = img[y:y+h, x:x+w]
            if imgCrop is None:
                imgCrop = roi
            else:
                imgCrop = np.hstack((imgCrop, roi))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    print("[INFO] Trying to access webcam...")

    cam = None
    for i in range(3):  # try camera indices 0, 1, 2
        temp_cam = cv2.VideoCapture(i)
        if temp_cam.read()[0]:
            cam = temp_cam
            print(f"[INFO] Webcam found at index {i}")
            break
        temp_cam.release()

    if cam is None:
        print("[ERROR] No working webcam found!")
        return

    print("[INFO] Place your hand inside the rectangles.")
    print("[INFO] Press 'C' to capture hand histogram, then 'S' to save and exit.")

    flagPressedC, flagPressedS = False, False
    hist = None
    imgCrop = None

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture frame. Exiting...")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                flagPressedC = True
                print("[INFO] Histogram captured.")
            else:
                print("[WARNING] No hand crop found to generate histogram.")
        elif keypress == ord('s') and flagPressedC:
            flagPressedS = True
            print("[INFO] Saving histogram...")
            break

        if not flagPressedS:
            imgCrop = build_squares(img)

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dst = cv2.filter2D(dst, -1, disc)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresholded Hand", thresh)

        cv2.imshow("Set Hand Histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        print("[SUCCESS] Histogram saved as 'hist'.")

get_hand_hist()
