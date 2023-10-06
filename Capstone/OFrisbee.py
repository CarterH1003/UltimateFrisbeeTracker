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
#show(hsv_image)

#whereFrisbee=np.uint8((hsv_image[::4,::4,0]>135)*(hsv_image[::4,::4,0]<145))*255
#show(whereFrisbee)

lower_orange = np.array([0, 80, 80])  # Lower bound of orange in HSV
upper_orange = np.array([30, 255, 255])  # Upper bound of orange in HSV

mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Apply a morphological operation (dilation) to the mask to connect neighboring orange pixels
kernel = np.ones((6, 6), np.uint8)  # You can adjust the kernel size if needed
eroded_mask = cv2.erode(mask, kernel, iterations=1)

# Apply the dilated mask to the original image to extract the orange regions with neighbors
orange_highlighted = cv2.bitwise_and(image, image, mask=eroded_mask)
show(orange_highlighted)

