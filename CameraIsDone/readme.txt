# CameraIsDone

## colorDetection.py

用于颜色检查

## ColorDetector.py

对输入图片检查红黄蓝绿四种颜色并画框

## Depth.py

获取深度，输出可视化深度图DepthMapVis和深度距离图DepthMap（作为对象属性）

## stereoconfig.py

包含标定好的相机参数

## test.py

用于测试颜色检查模块，经测试正常，若有问题可以检查路径

test_image.jpg为检查用图片

## Top.py

顶层文件，调用各模块在屏幕上实时展示颜色检查结果和深度结果