import cv2
import numpy as np

def show(img):
	# Display the image in a window named "Image"
	cv2.imshow('Image', img)

	# Wait until a key is pressed, then close the window
	cv2.waitKey(0)
	cv2.destroyAllWindows()


image = cv2.imread("tennis.jpg")


# ~ show(image[::4,::4,1])
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
whereBall=np.uint8((hsv_image[::4,::4,0]>37)*(hsv_image[::4,::4,0]<45))*255
show(whereBall)

