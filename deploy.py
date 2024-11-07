import math
import time
import os, glob
from pathlib import Path

import torch
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from threading import Thread

from utils.math_utils import project_gravity, wrap_to_pi
from robot import Robot, RobotObservation

device = "cpu"

# in sim order
clip_actions_high = [
    1.494,
    3.932,
    0.9260000000000002,
    1.894,
    3.932,
    0.9260000000000002,
    1.494,
    3.532,
    0.9260000000000002,
    1.894,
    3.532,
    0.9260000000000002,
]
clip_actions_low = [
    -1.894,
    -2.5260000000000002,
    -2.042,
    -1.494,
    -2.5260000000000002,
    -2.042,
    -1.894,
    -2.926,
    -2.042,
    -1.494,
    -2.926,
    -2.042,
]
th_clip_actions_high = torch.tensor(clip_actions_high, device=device)
th_clip_actions_low = torch.tensor(clip_actions_low, device=device)


def load_policy(root: Path, model_name: str = "go2_friction", device="cpu"):
    body_path = glob.glob(os.path.join(root, "models", model_name, "body*"))[0]
    adaptation_module_path = glob.glob(
        os.path.join(root, "models", model_name, "adaptation_module*")
    )[0]

    body = torch.jit.load(body_path, map_location=device)
    adaptation_module = torch.jit.load(adaptation_module_path, map_location=device)

    @torch.no_grad()
    def policy(stacked_history: torch.Tensor):
        # stacked_history: (H, d) = (15, 75)
        obs_history = stacked_history.reshape(1, 1125).squeeze().to(device)  # (1125, )
        latent = adaptation_module.forward(obs_history)  # (6, )
        action = body.forward(torch.cat((obs_history, latent), dim=-1))
        return action

    return policy


class DribbleEnv:
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
    hip_scale_reduction = torch.tensor(
        [0.5, 1, 1] * 4, dtype=torch.float32, device=device
    )
    torque_limit_clip = False
    set_small_cmd_to_zero = False

    def __init__(self, history_len: int, robot: Robot):
        assert history_len > 0

        self.history_len = history_len
        self.buffer = torch.zeros(
            history_len * 3, self.obs_dim, dtype=torch.float32, device=device
        )
        self.t = history_len

        self.action_t = torch.zeros(self.act_dim, dtype=torch.float32, device=device)
        self.action_t_minus1 = torch.zeros(
            self.act_dim, dtype=torch.float32, device=device
        )

        self.gait_index = 0.0
        self.yaw_init = 0.0

        self.robot = robot

        self.ball_position = None
        self.ball_position = [
            0.27402842,
            -0.04910268,
            0.0,
        ]  # TODO: init with None
        self.ball_subscriber = BallSubscriber(self)

    def observe(self):
        robot_obs = self.robot.get_obs()
        obs = self.make_obs(robot_obs)
        self.store_obs(obs)
        return self.buffer[self.t - self.history_len : self.t], robot_obs

    def store_obs(self, obs: torch.Tensor):
        h, buffer, t = self.history_len, self.buffer, self.t
        if t == buffer.shape[0]:
            buffer[:h] = buffer[t - h : t].clone()
            t = h
        buffer[t] = obs
        self.t = t + 1

    def advance(self, action: torch.Tensor):
        """action: in sim order"""
        self.action_t_minus1[:] = self.action_t
        self.action_t[:] = action

        # action_clipped = torch.clip(
        #     action, th_clip_actions_low, th_clip_actions_high
        # )  # TODO: this clip is important!
        action_clipped = action
        if self.torque_limit_clip:
            action_scaled = self.clip_by_torque_limit(
                action_clipped * self.action_scale * self.hip_scale_reduction
            )
        else:
            action_scaled = action_clipped * self.action_scale * self.hip_scale_reduction
        self.robot.set_act(action_scaled.tolist())
        self.gait_index = (self.gait_index + self.step_frequency * self.dt) % 1

    def clip_by_torque_limit(self, actions_scaled):
        """TODO: TO BE IMPLEMENTED"""
        
        p_limits_low = (-self.torque_limits) + self.d_gains * self.dof_vel_
        p_limits_high = (self.torque_limits) + self.d_gains * self.dof_vel_
        actions_low = (
            (p_limits_low / self.p_gains) - self.default_dof_pos + self.dof_pos_
        )
        actions_high = (
            (p_limits_high / self.p_gains) - self.default_dof_pos + self.dof_pos_
        )

        return torch.clip(actions_scaled, actions_low, actions_high)

    def make_obs(self, robot_obs: RobotObservation) -> torch.Tensor:
        ball_pos = torch.tensor(self.ball_position, device=device)
        projected_gravity = np.array(project_gravity(robot_obs.quaternion))
        projected_gravity = projected_gravity / np.linalg.norm(projected_gravity)
        commands = [
            # rocker x: left/right
            # rocker y: forward/backward
            robot_obs.ly * 2.0,  # x vel
            robot_obs.lx * 2.0,  # y vel
            0.0 * 0.25,  # yaw vel
            0.0 * 2.0,  # body height
            self.step_frequency,
            self.phase,
            self.offset,
            self.bound,
            self.duration,
            0.09 * 0.15,  # foot swing height
            0.0 * 0.3,  # pitch
            0.0 * 0.3,  # roll
            0.0,  # stance_width
            0.1 / 2,  # stance length
            0.01 / 2,  # unknown
        ]
        
        if self.set_small_cmd_to_zero and np.linalg.norm(commands[:2]) < 0.2:
            commands[:2] *= 0   # TODO: something error
            
        dof_pos = robot_obs.joint_position
        dof_vel = [v * 0.05 for v in robot_obs.joint_velocity]
        action = self.action_t
        last_action = self.action_t_minus1

        clock = self.clock()
        yaw = wrap_to_pi(robot_obs.yaw - self.yaw_init)
        timing = self.gait_index

        obs = torch.cat(
            [
                torch.tensor(
                    [
                        *ball_pos,
                        *projected_gravity,
                        *commands,
                        *dof_pos,
                        *dof_vel,
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                action,
                last_action,
                torch.tensor([*clock, yaw, timing], dtype=torch.float32, device=device),
            ]
        )
        return obs.to(device)

    def clock(self):
        return [
            math.sin(2 * math.pi * (self.gait_index + offset))
            for offset in self.foot_gait_offsets
        ]


class BallSubscriber(Node):
    def __init__(self, env: DribbleEnv):
        super().__init__("ball_subscriber")
        self.env = env
        self.subscription = self.create_subscription(
            Float32MultiArray, "ball_position", self.listener_callback, 0
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.env.ball_position = np.array(msg.data) / 1000.0
        assert len(self.env.ball_position) == 3


def log_to_file(obs, action, filename="./record/record.txt", mode="a"):
    obs = obs.cpu().numpy()
    action = action.cpu().numpy()
    sensor_info = {
        "ObjectSensor": obs[0:3],
        "OrientationSensor": obs[3:6],
        "RCSensor": obs[6:21],
        "JointPositionSensor": obs[21:33],
        "JointVelocitySensor": obs[33:45],
        "ActionSensor": obs[45:57],
        "ActionSensor_last": obs[57:69],
        "ClockSensor": obs[69:73],
        "YawSensor": obs[73:74],
        "TimingSensor": obs[74:75],
    }

    with open(filename, mode) as file:
        for sensor_name, sensor_values in sensor_info.items():
            file.write(f"{sensor_name}: {sensor_values}\n")
        file.write("-" * 20 + "\n")
        file.write(f"Action: {action}")
        file.write("-" * 40 + "\n")


def main(args):
    policy = load_policy(
        Path(__file__).resolve().parent, model_name=args.model_name, device=device
    )

    rclpy.init()
    robot = Robot()
    env = DribbleEnv(history_len=15, robot=robot)

    thread = Thread(target=rclpy.spin, args=(env.ball_subscriber,))
    thread.start()

    print("Robot initialized, press L1 to stand")
    while True:
        obs, robot_obs = env.observe()
        env.advance(torch.zeros(12, dtype=torch.float32, device=device))
        if robot_obs.L1:
            break
        time.sleep(env.dt)

    robot.to_stand()
    time.sleep(1)
    print("Robot ready, press L1 to start")
    while True:
        obs, robot_obs = env.observe()
        env.advance(torch.zeros(12, dtype=torch.float32, device=device))
        if robot_obs.L1:
            break
        time.sleep(env.dt)

    print("Robot started, press L2 to stop")

    env.yaw_init = robot_obs.yaw
    env.buffer[:, 73] = 0.0  # clear yaw sensor

    robot.to_run()

    benchmark = args.benchmark
    log = args.log
    assert not (benchmark and log), "Cannot benchmark and log at the same time"
    if benchmark:
        infer_time = []
    if log:
        i = 0
        all_obs = []
        all_actions = []

    while True:
        begin = time.perf_counter()

        obs, robot_obs = env.observe()
        action = policy(obs)
        env.advance(action)
        if robot_obs.L2:
            break

        if benchmark:
            end = time.perf_counter()
            print(end - begin)
            infer_time.append(end - begin)
            # time.sleep(max(0, begin + env.dt - end))

        if log:
            all_obs.append(obs[14, 21:33].cpu().numpy())
            all_actions.append(obs[14, 45:57].cpu().numpy())

            i += 1
            if i == 1:
                log_to_file(obs[-1], action, mode="w")
            else:
                log_to_file(obs[-1], action, mode="a")

        while time.perf_counter() < begin + env.dt:
            pass

    if benchmark:
        import matplotlib.pyplot as plt
        avg_infer_time = np.mean(infer_time)
        max_infer_time = np.max(infer_time)
        print(f"Average inference time: {avg_infer_time:.6f} seconds")
        print(f"Max inference time: {max_infer_time:.6f} seconds")
        plt.figure(figsize=(10, 6))
        plt.plot(infer_time, label="Inference time", color="blue")
        plt.xlabel('step')
        plt.ylabel('Inference time (s)')
        plt.title('Inference time over steps')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    if log:
        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        np.save("./record/all_obs", all_obs, allow_pickle=False)
        np.save("./record/all_actions", all_actions, allow_pickle=False)

    robot.to_damp()
    time.sleep(3)
    robot.to_relax()
    print("Robot relaxed, press L2 to quit")
    while True:
        obs, robot_obs = env.observe()
        if robot_obs.L2:
            break
    robot.stopped.set()
    robot.background_thread.join()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="go2_friction_2")
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
