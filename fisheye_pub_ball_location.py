import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import pickle

class BallPositionNode(Node):
    def __init__(self):
        super().__init__('ball_position_node')

        # 发布绿色物体的位置话题
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ball_position', 10)

        # 加载相机内参矩阵和畸变参数
        with open('./calib_pickle/calibration_params.pickle', 'rb') as f:
            calibration_params = pickle.load(f)
        self.K = calibration_params['K']
        self.D = calibration_params['D']

        # 加载外参
        with open('./calib_pickle/custom_coordinate_system.pickle', 'rb') as f:
            extrinsics = pickle.load(f)
        self.R = extrinsics['R']  # 旋转矩阵
        self.t = extrinsics['t']  # 平移向量

        # 逆变换以获得从相机坐标系到世界坐标系的转换
        self.R_inv = np.linalg.inv(self.R)
        self.t_inv = -self.R_inv @ self.t

        # 缩放因子
        self.scaling_factor = 2

        # 定义黄绿色的HSV颜色范围
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        # 打开摄像头
        self.cap = cv2.VideoCapture('/dev/video0')
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
            rclpy.shutdown()

        # 设置定时器，以周期性处理图像帧
        self.timer = self.create_timer(0.1, self.process_frame)  # 每0.1秒处理一次

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("无法读取帧")
            return

        # 转换到HSV颜色空间并应用颜色过滤
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓并计算中心点
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # 记录检测到的图像点
                image_points = np.array([[cx, cy]], dtype=np.float32)
                undistorted_points = cv2.fisheye.undistortPoints(image_points.reshape(-1, 1, 2), self.K, self.D, P=self.K)

                # 将图像点转换到相机坐标系下
                camera_point = np.array([undistorted_points[0][0][0], undistorted_points[0][0][1], 1]).reshape(3, 1)

                # 从相机坐标转换到世界坐标
                world_point = self.R_inv @ (camera_point - self.t_inv)
                world_point_2D = world_point.flatten()[:2] * self.scaling_factor  # 只取X和Y坐标并应用缩放

                # 发布位置
                msg = Float32MultiArray(data=[world_point_2D[0], world_point_2D[1], 0.0])
                self.publisher_.publish(msg)

                # 输出信息
                self.get_logger().info("绿色物体在地面坐标系中的二维位置（毫米）：{}".format(world_point_2D))
        else:
            # 如果没有检测到目标，发布默认值
            msg = Float32MultiArray(data=[0.0, 0.0, 0.0])
            self.publisher_.publish(msg)
            self.get_logger().info("未检测到目标")

    def release_resources(self):
        if self.cap.isOpened():
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
        node.release_resources()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
