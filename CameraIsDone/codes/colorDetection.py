import cv2
import numpy as np

class colorDetection():
    
    def __init__(self, in_goal_color: str):
        self.goal_color = in_goal_color
        self.bgr_img = []
        self.mask_open = []
        self.stats = []
        # colorDetection.color_detect(self, self.bgr_img, self.goal_color)
    # goal_color = 'yellow'

    def red_hsv(img):
        lower_hsv_1 = np.array([0, 100, 20])
        higher_hsv_1 = np.array([10, 255, 255])
        lower_hsv_2 = np.array([170, 100, 20])
        higher_hsv_2 = np.array([180, 255, 255])
        mask_1 = cv2.inRange(img, lower_hsv_1, higher_hsv_1)
        mask_2 = cv2.inRange(img, lower_hsv_2, higher_hsv_2)
        return mask_1 + mask_2

    def green_hsv(img):
        lower_hsv = np.array([40, 150, 20])
        higher_hsv = np.array([70, 255, 255])
        mask = cv2.inRange(img, lower_hsv, higher_hsv)
        return mask

    def blue_hsv(img):
        lower_hsv = np.array([95, 150, 20])
        higher_hsv = np.array([125, 255, 255])
        mask = cv2.inRange(img, lower_hsv, higher_hsv)
        return mask

    def yellow_hsv(img):
        lower_hsv = np.array([23, 100, 20])
        higher_hsv = np.array([35, 255, 255])
        mask = cv2.inRange(img, lower_hsv, higher_hsv)
        return mask

    def color_detect(self, bgr_img_1: np.ndarray) -> None:
        
        #cv2.imshow('bgr', bgr_img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #高斯模糊
        gs_img = cv2.GaussianBlur(bgr_img_1, (5, 5), 0)
        #bgr->hsv
        hsv_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2HSV)
        #寻找对应颜色
        if self.goal_color == 'red':
            mask_img = colorDetection.red_hsv(hsv_img)
        elif self.goal_color == 'green':
            mask_img = colorDetection.green_hsv(hsv_img)
        elif self.goal_color == 'blue':
            mask_img = colorDetection.blue_hsv(hsv_img)
        elif self.goal_color == 'yellow':
            mask_img = colorDetection.yellow_hsv(hsv_img)
        #cv2.imshow('mask', mask_img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #做开闭运算
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_close = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernal)
        self.mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernal)
        # cv2.imshow('mask', mask_open)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        #寻找连通区域
        #ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, ltype)
        #connectivity: 4或者8, 判断连通的像素点, 周围4像素或者8像素, 默认为8;
        #labels: 图像标记;
        #stats: [[x1, y1, width1, height1, area1], ...[xi, yi, widthi, heighti, areai]], 存放外接矩形和连通区域的面积信息,
        #    其中, stats[0]为背景;
        #centroids: [cen_x, cen_y], 质心的点坐标, 浮点类型
        ret, labels, self.stats, centroid = cv2.connectedComponentsWithStats(self.mask_open)
        # for i, stat in enumerate(stats):
        #     #绘制连通区域
        #     if (stat[-1] >= 100) & (i >= 1):
        #         cv2.rectangle(bgr_img_1, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (25, 25, 255), 3)
        self.bgr_img = bgr_img_1
        # cv2.imshow('bgr_img', bgr_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    img = cv2.imread("./StereoCalibration/images/imr_rectified1.jpg")
    blue_detector = colorDetection('blue')
    blue_detector.color_detect(img)
    cv2.imshow('bgr_img', blue_detector.bgr_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
        
    cv2.imshow('mask', blue_detector.mask_open)
    cv2.waitKey()
    cv2.destroyAllWindows()