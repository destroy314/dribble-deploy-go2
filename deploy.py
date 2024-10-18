import math
from pathlib import Path

import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from math_utils import project_gravity, wrap_to_pi
from robot import Robot, RobotObservation


def load_policy(root: Path):
    body = torch.jit.load("/home/unitree_go2/robo_soccer/tmp/legged_data/body_28400.jit", map_location='cpu')
    adaptation_module = torch.jit.load("/home/unitree_go2/robo_soccer/tmp/legged_data/adaptation_module_28400.jit", map_location='cpu')
    # body = torch.jit.load(root / 'body.jit', map_location='cpu')
    # adaptation_module = torch.jit.load(root / 'adaptation_module.jit', map_location='cpu')

    @torch.no_grad()
    def policy(stacked_history: torch.Tensor):
        # stacked_history: (H, d) = (15, 75)
        history = stacked_history.reshape(1, 1125)
        latent = adaptation_module(history)  # (1, 6)
        composed = torch.cat((history, latent), dim=-1)
        action = body(composed)
        return action[0]

    return policy


class RealtimeEnv:
    def observe(self): ...
    def advance(self, action): ...


class DribbleEnv(RealtimeEnv):
    obs_dim = 75
    act_dim = 12

    # gait type parameters
    phase = 0.5
    offset = 0.0
    bound = 0.0

    foot_gait_offsets = [phase + offset + bound, offset, bound, phase]

    duration = 0.5  # duration = stance / (stance + swing)
    step_frequency = 3.0

    control_decimation = 4
    simulation_dt = 0.005
    dt = control_decimation * simulation_dt

    action_scale = 0.25
    hip_scale_reduction = torch.tensor([0.5, 1, 1] * 4, dtype=torch.float32)

    def __init__(self, history_len: int, robot: Robot):
        assert history_len > 0

        self.history_len = history_len
        self.buffer = torch.zeros(history_len * 3, self.obs_dim, dtype=torch.float32)
        self.t = history_len

        self.action_t = torch.zeros(self.act_dim, dtype=torch.float32)
        self.action_t_minus1 = torch.zeros(self.act_dim, dtype=torch.float32)

        self.gait_index = 0.0

        self.yaw_init = 0.0

        self.robot = robot

        self.ball_position = [0.0, 0.0, 0.0]
        self.ball_subscriber = BallSubscriber(self)

    def observe(self):
        robot_obs = self.robot.get_obs()
        obs = self.make_obs(robot_obs)
        self.store_obs(obs)
        return self.buffer[self.t - self.history_len:self.t], robot_obs

    def advance(self, action: torch.Tensor):
        self.action_t_minus1[:] = self.action_t
        self.action_t[:] = action

        action_scaled = action * self.action_scale * self.hip_scale_reduction
        self.robot.set_act(action_scaled.tolist())
        self.gait_index = (self.gait_index + self.step_frequency * self.dt) % 1

    def store_obs(self, obs: torch.Tensor):
        h, buffer, t = self.history_len, self.buffer, self.t
        if t == buffer.shape[0]:
            buffer[:h] = buffer[t - h:t].clone()
            t = h
        buffer[t] = obs
        self.t = t + 1

    def make_obs(self, robot_obs: RobotObservation) -> torch.Tensor:
        ball_pos = self.ball_position
        projected_gravity = project_gravity(robot_obs.quaternion)
        commands = [
            # rocker x: left/right
            # rocker y: forward/backward
            robot_obs.ly * 2,   # x vel
            robot_obs.lx * 2,   # y vel
            0.0 * 0.25,         # yaw vel
            0.0 * 2,            # body height
            self.step_frequency,
            self.phase,
            self.offset,
            self.bound,
            self.duration,
            0.09 * 0.15,        # foot swing height
            0.0 * 0.3,          # pitch
            0.0 * 0.3,          # roll
            0.0,                # stance_width
            0.1 / 2,            # stance length
            0.01 / 2,           # unknown
        ]
        dof_pos = robot_obs.joint_position
        dof_vel = [v * 0.05 for v in robot_obs.joint_velocity]
        action = self.action_t
        last_action = self.action_t_minus1

        clock = self.clock()
        yaw = wrap_to_pi(robot_obs.yaw - self.yaw_init)
        timing = self.gait_index

        return torch.cat([
            torch.tensor([
                *ball_pos,
                *projected_gravity,
                *commands,
                *dof_pos,
                *dof_vel,
            ], dtype=torch.float32),
            action, last_action,
            torch.tensor([*clock, yaw, timing], dtype=torch.float32),
        ])

    def clock(self):
        return [
            math.sin(2 * math.pi * (self.gait_index + offset))
            for offset in self.foot_gait_offsets
        ]


class BallSubscriber(Node):
    def __init__(self, env: DribbleEnv):
        super().__init__('ball_subscriber')
        self.env = env
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ball_position',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.env.ball_position = msg.data


def main():
    import time
    policy = load_policy(Path(__file__).resolve().parent)
    robot = Robot()
    env = DribbleEnv(history_len=15, robot=robot)

    rclpy.init()
    rclpy.spin(env.ball_subscriber)

    robot.init()
    print('Robot initialized, press L1 to start')
    while True:
        obs, robot_obs = env.observe()
        env.advance(torch.zeros(12, dtype=torch.float32))
        if robot_obs.L1:
            break
        time.sleep(0.02)

    robot.to_stand()
    time.sleep(1)
    print('Robot started, press L2 to stop')
    env.yaw_init = robot_obs.yaw

    while True:
        begin = time.perf_counter()
        obs, robot_obs = env.observe()
        action = policy(obs)
        env.advance(action)
        if robot_obs.L2:
            break
        end = time.perf_counter()
        time.sleep(max(0, begin + 0.02 - end))

    robot.stopped.set()
    robot.background_thread.join()
    robot.to_damp()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
