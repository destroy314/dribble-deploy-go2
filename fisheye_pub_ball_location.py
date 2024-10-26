import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import cv2
import numpy as np
import pickle

class FisheyeCameraNode(Node):
    def __init__(self):
        super().__init__('fisheye_camera_node')
        
        # 发布相机和世界坐标的话题
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ball_position', 10)
        
        # 订阅外参的节点, 例如: 外参消息定义为 R 和 t
        self.subscription = self.create_subscription(
            String,
            'camera_extrinsics',
            self.extrinsic_callback,
            10)

        # 打开相机
        # pipeline = (
        #     "v4l2src device=/dev/video1 ! "
        #     "videoconvert ! appsink"
        # )
        pipeline = "/dev/video0"
        # self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.capture = cv2.VideoCapture(pipeline)
        if not self.capture.isOpened():
            self.get_logger().error("Failed to open camera")
        else:
            self.get_logger().info("Camera opened successfully")

        # 读取鱼眼相机的内参矩阵 K 和畸变参数
        with open('./calibration_params.pickle', 'rb') as f:
            calibration_params = pickle.load(f)
        self.K = calibration_params['K']
        self.D = calibration_params['D']
        self.DIM = calibration_params['DIM']

        self.get_logger().info('FisheyeCameraNode initialized.')

        # 设置定时器以周期性处理相机帧
        self.timer = self.create_timer(0.1, self.process_frame) # 每0.1秒处理一次

        # 初始化外参 (默认单位矩阵)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def extrinsic_callback(self, msg):
        """接收外参消息，更新旋转矩阵及平移向量"""
        try:
            # 解析外参，假设消息格式为字符串化的数组，这里为了简单直接使用eval
            extrinsic_data = eval(msg.data)
            self.R = np.array(extrinsic_data['R'])  # 旋转矩阵
            self.t = np.array(extrinsic_data['t'])  # 平移向量
            self.get_logger().info('Extrinsics updated: R and t received.')
        except (KeyError, SyntaxError) as e:
            self.get_logger().error(f"Failed to parse extrinsics: {e}")

    def undistort(self, img, K, D, DIM, scale=0.6):
        """自定义去畸变函数."""
        if img is None:
            return None
        dim1 = img.shape[:2][::-1]
        if dim1[0] / dim1[1] != DIM[0] / DIM[1]:
            img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)

        Knew = K.copy()
        if scale:
            Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def process_frame(self):
        retval, image = self.capture.read()
        if not retval:
            self.get_logger().error("Failed to read from camera.")
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = (40, 40, 40)
        upper_green = (80, 255, 255)
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        known_diameter = 0.18  # 物体的已知直径（米）

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cx, cy = x + w // 2, y + h // 2
            pixel_diameter = w

            if pixel_diameter > 0:
                # 计算 Z 轴上的距离
                focal_length = self.K[0, 0]
                distance = (known_diameter * focal_length) / pixel_diameter

                # 归一化坐标
                x_dis = (cx - image.shape[1] / 2)
                y_dis = (cy - image.shape[0] / 2)
                x_normal = x_dis / focal_length
                y_normal = y_dis / focal_length

                #球心与图像中心距离r
                r = np.sqrt(x_dis**2 + y_dis**2)

                # theta = r / f
                angle_theta = r / focal_length
                angle_theta = np.clip(angle_theta,0,np.pi/2)
                
                angle_phi = np.arctan2(y_normal, x_normal)

                # 转换为相机坐标系下的3D坐标
                X_cam = distance * np.sin(angle_theta) * np.cos(angle_phi)
                Y_cam = distance * np.sin(angle_theta) * np.sin(angle_phi)
                Z_cam = distance * np.cos(angle_theta)

                # 将相机坐标转换为世界坐标
                camera_coords = np.array([X_cam, Y_cam, Z_cam]).reshape(3, 1)
                world_coords = self.R @ camera_coords + self.t

                # 发布位置、距离以及世界坐标信息
                output_message = (
                    f"Camera Coordinates (X, Y, Z): ({X_cam:.2f}, {Y_cam:.2f}, {Z_cam:.2f}), "
                    f"World Coordinates (X_w, Y_w, Z_w): ({world_coords[0, 0]:.2f}, {world_coords[1, 0]:.2f}, {world_coords[2, 0]:.2f}), "
                    f"Distance: {distance:.2f}m, Angles (Theta, Phi): ({np.degrees(angle_theta):.2f}°, {np.degrees(angle_phi):.2f}°)"
                )

                self.publisher_.publish(Float32MultiArray(data=[-5*Y_cam,-5*X_cam,0.0]))
                self.get_logger().info(output_message)

                # 在图像上绘制
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
            else:
                self.get_logger().info("Object too small or too distant.")
                self.publisher_.publish(Float32MultiArray(data=[0.2,0.0,0.0]))
        else:
            self.get_logger().info("No object detected")
            self.publisher_.publish(Float32MultiArray(data=[0.2,0.0,0.0]))

        # cv2.imshow("Camera Output", image)
        # cv2.waitKey(1)

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    fisheye_camera_node = FisheyeCameraNode()

    try:
        rclpy.spin(fisheye_camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        fisheye_camera_node.capture.release()
        cv2.destroyAllWindows()
        fisheye_camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
