import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the video capture
cap = cv2.VideoCapture('LindenVid.mov', cv2.CAP_ANY)

frame_counter = 0
y1 = 0
y2 = 0
x1 = 0
x2 = 0

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the lower and upper bounds of the orange color in the HSV color space
lower_orange = np.array([0, 80, 80])  # Lower bound of orange in HSV
upper_orange = np.array([30, 255, 255])  # Upper bound of orange in HSV

# Create a kernel for dilation (adjust the size if needed)
kernel = np.ones((6, 6), np.uint8)

center_coordinates =[]

# Initialize an empty list to store FFT results
average_results = []

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

    orange_coords=np.transpose(np.nonzero(eroded_mask))

    if orange_coords.size:
        # Calculate the centroid of the orange coordinates
        center_x = np.mean(orange_coords[:, 1])
        center_y = np.mean(orange_coords[:, 0])
        center_coordinates.append((center_x, center_y))
        min_x = np.min(orange_coords[:, 1])
        max_x = np.max(orange_coords[:, 1])
        min_y = np.min(orange_coords[:, 0])
        max_y = np.max(orange_coords[:, 0])

        box_width = max_x - min_x
        box_height = max_y - min_y

         # Draw a box around the orange frisbee
        box_size = 45  # Adjust the size of the box as needed
        x1 = int(center_x - box_size / 2)
        y1 = int(center_y - box_size / 2)
        x2 = int(center_x + box_size / 2)
        y2 = int(center_y + box_size / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        average = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        average_results.append(average)
        #print(average)
    
    frame_counter += 1  # Increment the frame counter

    # Take the FFT of the average pixel values
    

    # Apply the dilated mask to the frame to extract the orange regions with neighbors
    orange_highlighted = cv2.bitwise_and(frame, frame, mask=eroded_mask)

    # Display the frame with the orange highlighted regions
    #cv2.imshow('Orange Highlighted', orange_highlighted)
    cv2.imshow('Orange Frisbee Boxed', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Plot the FFT results with frames on the x-axis
#plt.title('FFT Results')
frame_numbers = np.arange(1, frame_counter + 1)  # Create an array with frame numbers
    # plt.plot(frame_numbers[i], magnitude, 'bo')  # 'bo' for blue circles
plt.plot(list(range(len(average_results))), average_results)

plt.xlabel('Frame Number')
plt.ylabel('Log Magnitude')
plt.grid(True)
plt.show()