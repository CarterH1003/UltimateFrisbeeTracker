import cv2
import numpy as np

# Open the video capture
cap = cv2.VideoCapture('OFrisbeeVid.mov', cv2.CAP_ANY)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the lower and upper bounds of the orange color in the HSV color space
lower_orange = np.array([0, 80, 150])  # Lower bound of orange in HSV
upper_orange = np.array([100, 255, 255])  # Upper bound of orange in HSV

# Create a kernel for dilation (adjust the size if needed)
kernel = np.ones((6, 6), np.uint8)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    frame=cv2.blur(frame,(3,3))

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the orange regions in the frame
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Apply a morphological operation (dilation) to the mask to connect neighboring orange pixels
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

     # Apply the dilated mask to the frame to extract the orange regions with neighbors
    orange_highlighted = cv2.bitwise_and(frame, frame, mask=eroded_mask)

    # Display the frame with the orange highlighted regions
    cv2.imshow('Orange Highlighted', orange_highlighted)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()