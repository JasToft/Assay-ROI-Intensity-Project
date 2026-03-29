import cv2
import numpy as np

# Video settings
width, height = 400, 200
fps = 10
frames = 120

out = cv2.VideoWriter("synthetic_assay.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (width, height))

for i in range(frames):
    # White background
    frame = np.ones((height, width), dtype=np.uint8) * 255
    
    # Simulate gradual darkening
    intensity = int(255 * (1 - i / (frames - 1))) # Darken from white to black
    intensity = max(intensity, 0) # Limit minimum intensity

    # Draw test strip rectangle
    cv2.rectangle(frame, (150, 80), (250, 120),
                   intensity, -1)

    # Add some noise
    noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    out.write(frame)

out.release()
print("Video saved as synthetic_assay.mp4")
