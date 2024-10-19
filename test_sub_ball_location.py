import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import re

class CoordinateReceiver(Node):
    def __init__(self):
        super().__init__('coordinate_receiver_node')
        
        # 订阅 ball_position 话题获取相机和世界坐标信息
        self.subscription = self.create_subscription(
            String,
            'ball_position',
            self.listener_callback,
            10)
        
        # 输出一条初始化的日志信息
        self.get_logger().info('CoordinateReceiver initialized. Waiting for ball position data...')

    def listener_callback(self, msg):
        # 从消息字符串中提取出相机坐标 和 世界坐标
        message = msg.data
        
        # 定义提取相机坐标和世界坐标的正则表达式
        camera_coords_pattern = r"Camera Coordinates \(X, Y, Z\): \(([^,]+), ([^,]+), ([^)]+)\)"
        world_coords_pattern = r"World Coordinates \(X_w, Y_w, Z_w\): \(([^,]+), ([^,]+), ([^)]+)\)"
        
        # 使用正则表达式从字符串中提取相机坐标
        camera_coords_match = re.search(camera_coords_pattern, message)
        if camera_coords_match:
            X_cam = float(camera_coords_match.group(1))
            Y_cam = float(camera_coords_match.group(2))
            Z_cam = float(camera_coords_match.group(3))
            self.get_logger().info(f"Camera Coordinates: X = {X_cam:.2f}, Y = {Y_cam:.2f}, Z = {Z_cam:.2f}")
        else:
            self.get_logger().warn("Failed to parse camera coordinates.")
        
        # 使用正则表达式从字符串中提取世界坐标
        world_coords_match = re.search(world_coords_pattern, message)
        if world_coords_match:
            X_w = float(world_coords_match.group(1))
            Y_w = float(world_coords_match.group(2))
            Z_w = float(world_coords_match.group(3))
            self.get_logger().info(f"World Coordinates: X_w = {X_w:.2f}, Y_w = {Y_w:.2f}, Z_w = {Z_w:.2f}")
        else:
            self.get_logger().warn("Failed to parse world coordinates.")

def main(args=None):
    rclpy.init(args=args)

    # 创建并启动接收节点
    coordinate_receiver_node = CoordinateReceiver()

    try:
        rclpy.spin(coordinate_receiver_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 销毁节点并关闭 ROS2
        coordinate_receiver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()