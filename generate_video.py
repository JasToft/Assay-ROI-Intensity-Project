import cv2
import numpy as np

# Video settings
width, height = 400, 200
fps = 10
frames = 120

out =  cv2.VideoWriter("synthetic_assay.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (width, height))

for i in range(frames):
    # White background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Simulate gradual darkening
    intensity = int(255 - int(i / 1.5)) # Darken over time
    intensity = max(intensity, 50) # Limit minimum intensity

    # Draw test strip rectangle
    cv2.rectangle(frame, (150, 80), (250, 120),
                   (intensity, intensity, intensity), -1)

    # Add some noise
    noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
    frame = cv2.add(frame, noise)

    out.write(frame)

out.release()
print("Video saved as synthetic_assay.mp4")
