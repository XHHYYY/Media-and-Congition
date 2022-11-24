from colorDetection import colorDetection as CD
import numpy as np
import cv2

class ColorDetector():
    def __init__(self) -> None:
        self.Area = []
        self.RGBY = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    
    def Detect(self, img: np.ndarray, Depth: np.ndarray = None) -> np.ndarray:
        self.Area = img
        red_detector    = CD('red')
        green_detector  = CD('green')
        blue_detector   = CD('blue')
        yellow_detector = CD('yellow')
        
        for j, detector in enumerate([red_detector, green_detector, blue_detector, yellow_detector]):
            detector.color_detect(img)
            for i, stat in enumerate(detector.stats):
            #绘制连通区域
                if (stat[-1] >= 100) & (i >= 1):
                    cv2.rectangle(self.Area, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), self.RGBY[j], 3)
                    if Depth != None:
                        cv2.rectangle(Depth, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), self.RGBY[j], 3)

        return Depth