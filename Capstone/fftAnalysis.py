import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the video capture
cap = cv2.VideoCapture('LindenVid.mov', cv2.CAP_ANY)

frame_counter = 0

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the lower and upper bounds of the orange color in the HSV color space
lower_orange = np.array([0, 80, 80])  # Lower bound of orange in HSV
upper_orange = np.array([30, 255, 255])  # Upper bound of orange in HSV

# Create a kernel for dilation (adjust the size if needed)
kernel = np.ones((6, 6), np.uint8)

# Initialize an empty list to store the total number of orange pixels in each frame
orange_pixel_counts = []

# Read a fixed number of frames to reduce the computational load
num_frames = 1000
for _ in range(num_frames):
    # Read a frame from the video
    ret, frame = cap.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the orange regions in the frame
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Apply a morphological operation (dilation) to the mask to connect neighboring orange pixels
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Count the total number of orange pixels in the frame
    orange_pixel_count = np.sum(eroded_mask)

    # Append the total number of orange pixels to the list
    orange_pixel_counts.append(orange_pixel_count)

    frame_counter += 1  # Increment the frame counter

# Release the video capture
cap.release()

# Convert the list of pixel counts to a NumPy array
orange_pixel_counts = np.array(orange_pixel_counts)

# Compute FFT of a shorter segment of the total number of orange pixels
fft_data = np.fft.fft(orange_pixel_counts[:500])
magnitude_spectrum = np.abs(fft_data)

# Normalize FFT output for better visualization
magnitude_spectrum /= np.max(magnitude_spectrum)

# Plot the total number of orange pixels over frame number
plt.figure(figsize=(10, 8))

# Plot non-smoothed total number of orange pixels
plt.subplot(3, 1, 1)
plt.plot(np.arange(1, len(orange_pixel_counts) + 1), orange_pixel_counts)
plt.xlabel('Frame Number')
plt.ylabel('Total Orange Pixels')
plt.title('Total Orange Pixels in Video')
plt.grid(True)

# Plot FFT
plt.subplot(3, 1, 2)
plt.plot(np.arange(len(magnitude_spectrum)), magnitude_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Normalized Magnitude')
plt.title('Normalized FFT of Total Orange Pixels')
plt.grid(True)

plt.tight_layout()
plt.show()
