# -*- coding: UTF-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import pickle

class BallPositionNode(Node):
    def __init__(self):
        super().__init__('ball_position_node')

        # 发布球的位置话题
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ball_position', 10)

        # 加载相机内参矩阵和畸变参数
        with open('calibration_params.pickle', 'rb') as f:
            calibration_params = pickle.load(f)
        self.K = calibration_params['K']
        self.D = calibration_params['D']

        # 加载相机外参
        with open('custom_coordinate_system.pickle', 'rb') as f:
            extrinsics = pickle.load(f)
        self.R = extrinsics['R']  # 旋转矩阵
        self.t = extrinsics['t']  # 平移向量

        # 打开摄像头
        self.cap = cv2.VideoCapture('/dev/video0')
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
            rclpy.shutdown()

        # 已知球的直径（米）
        self.ball_diameter = 0.18

        # HSV颜色空间中的蓝色范围
        self.lower_blue = np.array([40,40,40])
        self.upper_blue = np.array([80, 255, 255])

        # 设置定时器，以周期性处理图像帧
        self.timer = self.create_timer(0.1, self.process_frame)  # 每0.1秒处理一次

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("无法读取帧")
            return

        # 转换为HSV颜色空间并应用颜色过滤 a绿色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓并计算外接矩形
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w / 2, y + h / 2  # 矩形中心点

            # 估算深度信息
            focal_length = self.K[0, 0]  # 使用相机矩阵中的焦距
            depth_estimate = (focal_length * self.ball_diameter) / h  # 估算物体深度

            # 将中心点坐标转换为图像中心为原点的坐标
            img_center_x = frame.shape[1] / 2
            img_center_y = frame.shape[0] / 2
            normalized_x = (center_x - img_center_x) * depth_estimate / focal_length
            normalized_y = (center_y - img_center_y) * depth_estimate / focal_length

            # 地面上的二维坐标（假设 z=0 平面）
            world_point_2D = np.array([normalized_x * 1000, normalized_y * 1000])  # 转换为毫米单位

            # 发布位置
            msg = Float32MultiArray(data=[world_point_2D[0], world_point_2D[1], 0.0])
            self.publisher_.publish(msg)

            # 输出信息
            self.get_logger().info("绿色球在地面坐标系中的二维位置（毫米）：{}".format(world_point_2D))

            cv2.imshow("cam",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        else:
            # 如果没有检测到目标，发布默认值
            msg = Float32MultiArray(data=[0.0, 0.0, 0.0])
            self.publisher_.publish(msg)
            self.get_logger().info("未检测到目标")

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = BallPositionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
