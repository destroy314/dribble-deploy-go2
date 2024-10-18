import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VisualizeNode(Node):
    def __init__(self):
        super().__init__('visualize_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ball_position',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.fig, self.ax = plt.subplots()
        self.robot = patches.Rectangle((-0.5, -0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        self.ball = patches.Circle((0, 0), 0.1, linewidth=1, edgecolor='b', facecolor='none')
        self.arrow = None
        self.ax.add_patch(self.robot)
        self.ax.add_patch(self.ball)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        plt.ion()
        plt.show()

    def listener_callback(self, msg):
        ball_pos = msg.data[:2]
        ball_vel = msg.data[2:]
        self.ball.set_center(ball_pos)
        if self.arrow:
            self.arrow.remove()
        self.arrow = patches.FancyArrow(ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1], head_width=0.05, head_length=0.1, fc='g', ec='g')
        self.ax.add_patch(self.arrow)
        plt.draw()

def main(args=None):
    rclpy.init(args=args)
    visualize_node = VisualizeNode()
    rclpy.spin(visualize_node)
    visualize_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
