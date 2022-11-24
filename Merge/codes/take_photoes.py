import cv2
cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cam0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cam2.set(3, 1920)
# cam2.set(4, 1080)

# cam0.set(3, 1920)
# cam0.set(4, 1080)

index = 1
baselength = 8
while (True):
    _, frame2 = cam2.read()
    _, frame0 = cam0.read()
    cv2.imshow('Capture2', frame2)
    cv2.imshow('Capture0', frame0)
    k = cv2.waitKey(10)
    if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        # cv2.imwrite('./StereoCalibration/images/LEFT/b8/' + str(index) + ' baseline = ' + str(baselength) + '.jpg', frame0)
        # cv2.imwrite('./StereoCalibration/images/RIGHT/b8/' + str(index) + ' baseline = ' + str(baselength) + '.jpg', frame2)
        cv2.imwrite('StereoCalibration/images/LEFT/Depth' + str(index) + '.jpg', frame0)
        cv2.imwrite('StereoCalibration/images/RIGHT/Depth' + str(index) + '.jpg', frame2)
        index += 1
    elif k == ord('q'):  # 按下q键，程序退出
        break
cam2.release() # 释放摄像头
cam0.release() # 释放摄像头
cv2.destroyAllWindows()# 释放并销毁窗口
