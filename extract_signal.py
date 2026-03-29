import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_signal(video_path, roi=None):
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        # Select ROI only if not provided
        if roi is None:
            roi = cv2.selectROI("Select ROI", frame)
            cv2.destroyAllWindows()

        x, y, w, h = roi
        signal = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi_frame = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            signal.append(255 - np.mean(gray))

        return np.array(signal), roi
    finally:
        cap.release()
if __name__ == "__main__":
    video_files = [
        "pos1.mp4",
        "pos2.mp4",
        "neg1.mp4",
        "neg2.mp4"
    ]

    all_raw = []
    roi = None

    # --- Pass 1: collect all raw signals ---
    for video in video_files:
        print(f"Processing {video}...")
        raw, roi = extract_signal(video, roi)
        all_raw.append((video, raw))

    # --- Global normalization ---
    global_min = min(raw.min() for _, raw in all_raw)
    global_max = max(raw.max() for _, raw in all_raw)

    all_signals = []
    for video, raw in all_raw:
        norm = (raw - global_min) / (global_max - global_min)
        smooth = pd.Series(norm).rolling(5).mean().fillna(0)

        all_signals.append((video, norm, smooth))

        # Save per video
        df = pd.DataFrame({
            "frame": np.arange(len(raw)),
            "raw": raw,
            "normalized": norm,
            "smoothed": smooth
        })
        df.to_csv(f"{video}_signal.csv", index=False)

    # --- Plot all signals together ---
    plt.figure(figsize=(10, 5))

    for video, norm, smooth in all_signals:
        plt.plot(norm, label=f"{video} (norm)", alpha=0.6)
        plt.plot(smooth, linestyle='--', label=f"{video} (smooth)")

    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Signal Comparison Across Videos")
    plt.legend()
    plt.grid()
    plt.show()