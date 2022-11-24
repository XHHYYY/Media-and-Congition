import StereoCalibration.codes.colorDetection as CD
import numpy as np
import cv2

class ColorDetector():
    def __init__(self) -> None:
        self.Area = []
        self.RGBY = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    
    def Detect(self, img: np.ndarray) -> None:
        self.Area = img
        red_detector = CD.colorDetection('red')
        green_detector = CD.colorDetection('green')
        blue_detector = CD.colorDetection('blue')
        yellow_detector = CD.colorDetection('yellow')
        
        for detector in [red_detector, green_detector, blue_detector, yellow_detector]:
            detector.color_detect(img)
            for i, stat in enumerate(detector.stats):
            #绘制连通区域
                if (stat[-1] >= 100) & (i >= 1):
                    cv2.rectangle(self.Area, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), self.RGBY[i], 3)
                    
