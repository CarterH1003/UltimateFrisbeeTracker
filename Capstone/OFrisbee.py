import cv2
import numpy as np

def show(img):
    # Display the image in a window named "Image"
	cv2.imshow('Image', img)

	# Wait until a key is pressed, then close the window
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
image = cv2.imread("OFrisbee.jpeg")
#show(image)
image=cv2.blur(image,(3,3))
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
show(hsv_image)

center_coordinates =[]

#whereFrisbee=np.uint8((hsv_image[::4,::4,0]>135)*(hsv_image[::4,::4,0]<145))*255
#show(whereFrisbee)

lower_orange = np.array([0, 80, 80])  # Lower bound of orange in HSV
upper_orange = np.array([30, 255, 255])  # Upper bound of orange in HSV

mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Apply a morphological operation (dilation) to the mask to connect neighboring orange pixels
kernel = np.ones((6, 6), np.uint8)  # You can adjust the kernel size if needed
eroded_mask = cv2.erode(mask, kernel, iterations=1)
orange_coords=np.transpose(np.nonzero(eroded_mask))

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
box_size = 50  # Adjust the size of the box as needed
x1 = int(center_x - box_size / 2)
y1 = int(center_y - box_size / 2)
x2 = int(center_x + box_size / 2)
y2 = int(center_y + box_size / 2)
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box


# Apply the dilated mask to the original image to extract the orange regions with neighbors
orange_highlighted = cv2.bitwise_and(image, image, mask=eroded_mask)
show(image)

