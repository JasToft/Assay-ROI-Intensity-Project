import cv2
import numpy as np

def generate_synthetic_video(filename, speed):
    # Video settings
    width, height = 400, 200
    fps = 10
    frames = 120

    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for i in range(frames):
        # White background
        frame = np.ones((height, width), dtype=np.uint8) * 255

        # Simulate gradual darkening
        intensity = int(255 * (1 - (i * speed) / (frames - 1)))
        intensity = max(intensity, 0)

        # Draw filled test strip (changes over time)
        cv2.rectangle(frame, (150, 80), (250, 120), intensity, -1)

        # Always draw black outline
        cv2.rectangle(frame, (150, 80), (250, 120), 0, 2)

        # Add noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        out.write(frame)

    out.release()
    print(f"Video saved as {filename}")


if __name__ == "__main__":
    generate_synthetic_video("pos1.mp4", 1.0)
    generate_synthetic_video("pos2.mp4", 0.8)

    generate_synthetic_video("neg1.mp4", 0.05)
    generate_synthetic_video("neg2.mp4", 0.02)