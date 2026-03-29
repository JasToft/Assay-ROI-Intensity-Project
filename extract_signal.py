import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    signal.append(255 - np.mean(gray))

cap.release()

signal = np.array(signal)

# Normalize signal
signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Smooth signal with moving average
signal_smooth = pd.Series(signal_norm).rolling(5).mean()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(signal_norm, label="Normalized Signal")
plt.plot(signal_smooth, label="Smoothed Signal", linewidth=2)
plt.xlabel("Frame")
plt.ylabel("Intensity")
plt.title("Signal Intensity Over Time")
plt.legend()

plt.show()

