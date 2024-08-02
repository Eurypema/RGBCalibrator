# RGBCalibrator.py

import cv2
import numpy as np

# Initialize accumulators for mean and M2 (sum of squares of differences from the mean)
mean_r, mean_g, mean_b = 0.0, 0.0, 0.0
M2_r, M2_g, M2_b = 0.0, 0.0, 0.0
num_pixels = 0
num_frames = 100

# Open the webcam
cap = cv2.VideoCapture(0)

# Sample frames
for _ in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB; OpenCV represents images in BGR by default, but PyTorch expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Flatten the image into a 1D array of pixels
    pixels = frame_rgb.reshape(-1, 3)

    # Update the total number of pixels
    num_pixels += pixels.shape[0]

    # Update mean and M2 using Welford's method (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
    for i in range(pixels.shape[0]):
        delta_r = pixels[i, 0] - mean_r # delta_r is the difference between the current pixel's red value and the current red value mean
        mean_r += delta_r / num_pixels # Updates red value mean to ensure gradual mean convergence to true mean as pixels are processed
        M2_r += delta_r * (pixels[i, 0] - mean_r) # Accumulates sum of squares of differences of mean, used to calculate variance 
        
        delta_g = pixels[i, 1] - mean_g
        mean_g += delta_g / num_pixels
        M2_g += delta_g * (pixels[i, 1] - mean_g)
        
        delta_b = pixels[i, 2] - mean_b
        mean_b += delta_b / num_pixels
        M2_b += delta_b * (pixels[i, 2] - mean_b)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Calculate variance and standard deviation
var_r = M2_r / num_pixels
var_g = M2_g / num_pixels
var_b = M2_b / num_pixels

stddev_r = np.sqrt(var_r)
stddev_g = np.sqrt(var_g)
stddev_b = np.sqrt(var_b)

print(f"Mean (R, G, B): ({mean_r:.3f}, {mean_g:.3f}, {mean_b:.3f})")
print(f"Variance (R, G, B): ({var_r:.3f}, {var_g:.3f}, {var_b:.3f})")
print(f"Standard Deviation (R, G, B): ({stddev_r:.3f}, {stddev_g:.3f}, {stddev_b:.3f})")
