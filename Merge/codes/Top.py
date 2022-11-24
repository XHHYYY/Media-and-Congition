import cv2
from StereoCalibration.codes.stereoconfig import stereoCamera
import scipy.io as scio
from ColorDetector import ColorDetector
from Depth import Depth as DP

index = 1
cam0 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
config = stereoCamera()

frame0 = []
frame2 = []

Depth = DP()
CDor = ColorDetector()

while (True):

    _, frame2 = cam2.read()
    _, frame0 = cam0.read()

    Depth.get_Depth(frame2, frame0, 'Q')
    CDor.Detect(frame2)
    cv2.imshow('Depth Map', Depth.depthMapVis)
    cv2.imshow('Color Detected', CDor.Area)
    
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('./Merge/images/color_Detected' + str(index) + '.jpg', CDor.Area)
        scio.savemat('./Merge/images/Depth_map' + str(index) + '.mat', {'Depth': Depth.DepthMap})
        index += 1
cam2.release() # 释放摄像头
cam0.release() # 释放摄像头
cv2.destroyAllWindows()# 释放并销毁窗口
