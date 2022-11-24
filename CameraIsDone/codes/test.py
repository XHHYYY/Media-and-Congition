from ColorDetector import ColorDetector
import cv2

img = cv2.imread('./CameraIsDone/test_image.jpg')
CDr = ColorDetector()

CDr.Detect(img)
cv2.imshow('color_test', CDr.Area)
cv2.waitKey()
cv2.destroyAllWindows()