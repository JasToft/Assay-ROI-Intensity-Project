import cv2
import numpy as np

cap = cv2.VideoCapture("synthetic_assay.mp4")

# Jump to last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
ret, frame = cap.read()

roi = cv2.selectROI("Select ROI", frame)
cv2.destroyAllWindows()

x, y, w, h = roi

signal = []

# Reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    signal.append(np.mean(gray))

cap.release()