import cv2
from stereoconfig import stereoCamera
import scipy.io as scio
from ColorDetector import ColorDetector
from Depth import Depth as DP

inverse = False
color_The_Depth = False

index = 1
if inverse:
    cam0 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cam2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cam0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
config = stereoCamera()

frame0 = []
frame2 = []

Depth = DP()
CDor = ColorDetector()

while (True):

    _, frame2 = cam2.read() # 左图，如果发现反了就把inverse置True
    _, frame0 = cam0.read() # 右图

    Depth.get_Depth(frame2, frame0, 'Q') # 使用Q或config或二者均值计算深度，Q效果应该更好
    if color_The_Depth:
        colored_Depth = CDor.Detect(frame2, Depth.depthMapVis) # 同时在深度图上画框
        cv2.imshow('Depth Map', colored_Depth)
    else:
        CDor.Detect(frame2, None)
        cv2.imshow('Depth Map', Depth.depthMapVis)
    cv2.imshow('Color Detected', CDor.Area)
    
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('./CameraIsDone/RecordedFiles/images/color_Detected' + str(index) + '.jpg', CDor.Area)
        scio.savemat('./CameraIsDone/RecordedFiles/DepthMaps/Depth_map' + str(index) + '.mat', {'Depth': Depth.DepthMap})
        index += 1
cam2.release() # 释放摄像头
cam0.release() # 释放摄像头
cv2.destroyAllWindows()# 释放并销毁窗口
