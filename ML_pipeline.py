import numpy as np
from extract_signal import extract_signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report


# Feature extraction
def extract_features(signal):
    return [
        np.max(signal),                  # peak intensity
        signal[-1] - signal[0],          # total change
        np.mean(signal),                 # average intensity
        np.max(np.diff(signal))          # max slope
    ]


if __name__ == "__main__":
    video_files = [
        "pos1.mp4",
        "pos2.mp4",
        "neg1.mp4",
        "neg2.mp4"
    ]

    all_raw = []
    roi = None

    print("Extracting signals...\n")

    # Pass 1: get raw signals
    for video in video_files:
        raw, roi = extract_signal(video, roi)
        all_raw.append((video, raw))

    # Global normalization
    global_min = min(raw.min() for _, raw in all_raw)
    global_max = max(raw.max() for _, raw in all_raw)

    X = []
    y = []

    print("Building dataset...\n")

    for video, raw in all_raw:
        norm = (raw - global_min) / (global_max - global_min)

        features = extract_features(norm)
        X.append(features)

        label = 1 if "pos" in video else 0
        y.append(label)

        print(f"{video}")
        print(f"  features: {np.round(features, 3)}")
        print(f"  label: {label}\n")

    X = np.array(X)
    y = np.array(y)

    # Train and evaluate with leave-one-out CV
    print("Training model (leave-one-out CV)...\n")

    model = LogisticRegression()
    preds = cross_val_predict(model, X, y, cv=LeaveOneOut())

    print("Predictions:", preds)
    print("True labels:", y)
    print()
    print(classification_report(y, preds, target_names=["negative", "positive"]))