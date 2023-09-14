import cv2
import numpy as np

def show(img):
    # Display the image in a window named "Image"
	cv2.imshow('Image', img)

	# Wait until a key is pressed, then close the window
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
image = cv2.imread("SFrisbee.jpeg")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show(hsv_image)

whereFrisbee=np.uint8((hsv_image[::4,::4,0]>252)*(hsv_image[::4,::4,0]<255))*255
(whereFrisbee)

