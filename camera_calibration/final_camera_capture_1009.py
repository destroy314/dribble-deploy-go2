import cv2
import numpy as np
import glob

# 设置棋盘格尺寸
CHECKERBOARD = (5, 8)  # 行和列的内角点数目

# 读取摄像头内容
capture = cv2.VideoCapture('/dev/video2')  # 打开笔记本内置摄像头

while capture.isOpened():  # 笔记本内置摄像头被打开后
    retval, image = capture.read()  # 从摄像头中实时读取视频
    if not retval:
        print("Failed to read from camera.")
        break
    
    cv2.imshow("Video", image)  # 在窗口中显示读取到的视频
    key = cv2.waitKey(1)  # 窗口的图像刷新时间为1毫秒

    if key == ord('n'):  # 如果按下 'n' 键
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 查找棋盘格角点

        if ret:
            # 如果找到角点，绘制角点
            #cv2.drawChessboardCorners(image, CHECKERBOARD, corners, ret)
            #cv2.imshow("Detected Corners", image)  # 显示检测到角点的图像
            
            # 保存图片
            photo_count = len(glob.glob('/home/hgd/data/calibration_1009/photo_*.jpg'))  # 统计已有图片数量
            cv2.imwrite(f'/home/hgd/data/calibration_1009/photo_{photo_count + 1}.jpg', image)  # 保存图片
            print(f'Photo {photo_count + 1} saved.')
        else:
            print('No corners found in the current frame.')

    elif key == ord('e'):  # 如果按下 'e' 键
        break  # 结束拍摄

# 释放摄像头
capture.release()  # 关闭笔记本内置摄像头
cv2.destroyAllWindows()  # 销毁显示摄像头视频的窗口

