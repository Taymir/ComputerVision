#coding=utf-8
import imutils
import cv2

image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

(B,G,R) = image[100,50]
print("R={}, G={}, B={}".format(R, G, B))

#cv2.imshow("Image", image)

roi = image[60:160, 320:420]
#cv2.imshow("ROI", roi)

#r = 300.0 / w
#dim = (300, int(h * r))
#resized = cv2.resize(image, dim)
resized = imutils.resize(image, width=300)
#cv2.imshow("Resized", resized)

#center = (w // 2, h // 2)
#M = cv2.getRotationMatrix2D(center, 30, 1.0)
#rotated = cv2.warpAffine(image, M, (w,h))
rotated = imutils.rotate_bound(image, 30)
#cv2.imshow("rotation", rotated)

blurred = cv2.GaussianBlur(image, (11, 11), 0)
#cv2.imshow("blurred", blurred)

output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0,0,255), 2)
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
#cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
#cv2.line(output, (60, 200), (400, 20), (0, 0, 255), 5)
cv2.putText(output, "OpenCV + Jurassic Park!!!", (w // 2, h - 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("rectangle", output)


cv2.waitKey(0)
