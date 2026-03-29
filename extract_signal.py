import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_signal(video_path, roi=None):
    cap = cv2.VideoCapture(video_path)

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

    cap.release()

    signal = np.array(signal)

    # Normalize
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    return signal, signal_norm, roi


if __name__ == "__main__":
    video_files = [
        "pos1.mp4",
        "pos2.mp4",
        "neg1.mp4",
        "neg2.mp4"
    ]

    all_signals = []
    roi = None

    for video in video_files:
        print(f"Processing {video}...")

        raw, norm, roi = extract_signal(video, roi)
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

    colors = {
    "pos": "green",
    "neg": "red"
}

    for video, norm, smooth in all_signals:
        color = "green" if "pos" in video else "red"
        plt.plot(norm, color=color, alpha=0.5)


    # Plot all signals together
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