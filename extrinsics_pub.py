import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np

class CameraExtrinsicsPublisher(Node):
    def __init__(self):
        super().__init__('camera_extrinsics_publisher')

        # 创建一个发布者，发布外参消息
        self.publisher_ = self.create_publisher(String, 'camera_extrinsics', 10)

        # 设置外参，假设旋转矩阵 R 和平移向量 t 以已知形式存在
        self.R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.t = np.array([[0.0], [0.0], [0.0]])

        # 打印传到ROS log中
        self.get_logger().info('Initialized Camera Extrinsics Publisher.')

        # 创建一个以固定间隔发布的定时器
        self.timer = self.create_timer(1.0, self.publish_extrinsics)  # 每 1 秒钟发布一次外参

    def publish_extrinsics(self):
        # 将 R 和 t 转换为字典（字符串）
        extrinsic_message = {
            "R": self.R.tolist(),   # 将 numpy 矩阵转换为Python列表
            "t": self.t.tolist()    # 同上
        }
        
        # 将字典转为字符串格式以便通过 String 消息传递
        msg = String()
        msg.data = str(extrinsic_message)

        # 发布消息
        self.publisher_.publish(msg)

        # 打印日志，显示发布内容
        self.get_logger().info(f'Published Extrinsics: {msg.data}')

def main(args=None):
    rclpy.init(args=args)

    # 创建并运行外参发布节点
    extrinsics_publisher = CameraExtrinsicsPublisher()

    try:
        rclpy.spin(extrinsics_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # 销毁节点并关闭 ROS2 通信
        extrinsics_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()