import imutils
import cv2
import argparse
import numpy as np

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

h = 400

ratio = image.shape[0] / h
image = imutils.resize(image, height = h)
orig = image.copy()
cv2.imshow("original", orig)
cv2.waitKey(0)

#adjusted = adjust_gamma(image, 2.0)
adjusted = np.zeros(image.shape, image.dtype)
alpha = 1 #contrast
beta = 0    #brightness
#for y in range(image.shape[0]):
#    for x in range(image.shape[1]):
#        for c in range(image.shape[2]):
#            adjusted[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#cv2.imshow("adjusted", adjusted)
#cv2.waitKey(0)

gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
blur = 5
gray = cv2.GaussianBlur(gray, (blur, blur), 0)
#gray = cv2.medianBlur(gray, 7)

#(T, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1, 12)
#gray = cv2.bilateralFilter(gray, 9, 9, 9)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
canny_p = 10
edged = cv2.Canny(gray, canny_p, int(canny_p/2))

cv2.imshow("edged", edged)
cv2.waitKey(0)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, int(h/2), param1=canny_p, minRadius=int(h*0.1), maxRadius=int(h*0.9))
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")

	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(orig, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(orig, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), -1)

	# show the output image
	cv2.imshow("output", orig)
	cv2.waitKey(0)

#(T, thresh) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)

#cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# loop over our contours
#for c in cnts:
	# approximate the contour
#	peri = cv2.arcLength(c, True)
#	approx = cv2.approxPolyDP(c, 0.015 * peri, True)

#cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
#cv2.imshow("Game Boy Screen", image)
#cv2.waitKey(0)
