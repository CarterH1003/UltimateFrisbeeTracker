import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the video capture
cap = cv2.VideoCapture('LindenVid.mov', cv2.CAP_ANY)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the lower and upper bounds of the orange color in the HSV color space
lower_orange = np.array([0, 80, 80])  # Lower bound of orange in HSV
upper_orange = np.array([30, 255, 255])  # Upper bound of orange in HSV

# Create a kernel for dilation (adjust the size if needed)
kernel = np.ones((6, 6), np.uint8)

# Get total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the frame index to the middle of the video
desired_frame_index = total_frames // 2

# Set the current frame index
current_frame_index = 0

while current_frame_index < desired_frame_index:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Increment the current frame index
    current_frame_index += 1

# If we successfully reached the desired frame
if current_frame_index == desired_frame_index:
    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the orange regions in the frame
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Apply a morphological operation (dilation) to the mask to connect neighboring orange pixels
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Display the frame after masking
    plt.imshow(cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=eroded_mask), cv2.COLOR_BGR2RGB))
    plt.title('Masked Frame from the Middle of the Video')
    plt.axis('off')  # Turn off axis
    plt.show()

# Release the video capture
cap.release()
