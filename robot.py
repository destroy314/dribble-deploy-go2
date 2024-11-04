import importlib.machinery
import importlib.util
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Policy output order: FL, FR, RL, RR
# Unitree control order: FR, FL, RR, RL
id_to_real_index = {
    "FR_0": 0,
    "FR_1": 1,
    "FR_2": 2,
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

real_index_to_id = {v: k for k, v in id_to_real_index.items()}

name_to_id = {
    "FL_hip_joint": "FL_0",
    "FL_thigh_joint": "FL_1",
    "FL_calf_joint": "FL_2",
    "FR_hip_joint": "FR_0",
    "FR_thigh_joint": "FR_1",
    "FR_calf_joint": "FR_2",
    "RL_hip_joint": "RL_0",
    "RL_thigh_joint": "RL_1",
    "RL_calf_joint": "RL_2",
    "RR_hip_joint": "RR_0",
    "RR_thigh_joint": "RR_1",
    "RR_calf_joint": "RR_2",
}

id_to_name = {v: k for k, v in name_to_id.items()}

sim_index_to_name = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

name_to_sim_index = {name: i for i, name in enumerate(sim_index_to_name)}

name_to_q0 = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

LegID = {
    "FR_0": 0,  # Front right hip
    "FR_1": 1,  # Front right thigh
    "FR_2": 2,  # Front right calf
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

PosStopF = 2.146e9
VelStopF = 16000.0


num_joints = len(sim_index_to_name)

# sim index to real index mapping is for real-to-sim conversion
# for sim_idx, real_idx in enumerate(sim_index_to_real_index):
#     assert real_joints[real_idx] == sim_joints[sim_idx]
sim_idx_to_real_idx = [id_to_real_index[name_to_id[name]] for name in sim_index_to_name]

# real index to sim index mapping is for sim-to-real conversion
# even if dict is ordered in Python 3.7 or later, it is not sorted by its key
# thus, we need to use `real_index_to_id[i] for i in range(num_joints)`
# instead of `for id in real_index_to_id.values()`
real_idx_to_sim_idx = [
    name_to_sim_index[id_to_name[real_index_to_id[i]]] for i in range(num_joints)
]
# [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

q0_sim = [name_to_q0[name] for name in sim_index_to_name]
q0_real = [name_to_q0[id_to_name[real_index_to_id[i]]] for i in range(num_joints)]


@dataclass
class RobotObservation:
    joint_position: "list[float]"
    joint_velocity: "list[float]"
    gyroscope: "list[float]"
    quaternion: "list[float]"
    roll: float
    pitch: float
    yaw: float
    lx: float
    ly: float
    rx: float
    ry: float
    L1: bool
    L2: bool


class Robot(Node):
    def __init__(self):
        super().__init__("robot_node")
        # pybind11 will convert std::array into python list
        self.motor_state_real = None  # std::array<MotorState, 20>

        self.imu = None
        # self.rc = None
        self.L1 = False
        self.L2 = False
        self.ball_vel = [0, 0, 0, 0]

        self.kp = 20.0
        self.kd = 0.5

        # the struct module does have compiled format cache
        # reuse the Struct object explicitly for clarity, not performance
        self.rocker_struct = struct.Struct("@5f")

        self.Δq_real = [None for _ in range(12)]  # in real order
        self.q_setted = False
        self.to_damp()

        self.stopped = Event()

        self.publisher_ = self.create_publisher(Float32MultiArray, "ball_velocity", 10)

        # ChannelFactoryInitialize(0, "eth0")
        ChannelFactoryInitialize(0)
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self._recv_cb, 10)

        sub1 = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        sub1.Init(self.WirelessControllerHandler, 10)

        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()
        self.pub = pub

        self.crc = CRC()

        while not self.q_setted:
            time.sleep(0.01)
        self.background_thread = Thread(target=self._send_loop, daemon=True)
        self.background_thread.start()

    def WirelessControllerHandler(self, msg: WirelessController_):
        self.L1 = True if msg.keys == 2 else False
        self.L2 = True if msg.keys == 32 else False
        self.ball_vel = [msg.lx, msg.ly, msg.rx, msg.ry]
        # print(self.ball_vel)
        self.publish_ball_speed()

    def _recv_cb(self, msg: LowState_):
        self.motor_state_real = msg.motor_state
        if self.q_setted == False:
            for i in range(12):
                self.Δq_real[i] = self.motor_state_real[i].q - q0_real[i]
            self.q_setted = True
        self.imu = msg.imu_state
        # self.rc = msg.wireless_remote
        # for byte in self.rc:
        #     print(f'{byte:08b}', end=' ')
        # print()

    def _send_loop(self):
        # return
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            cmd.motor_cmd[i].q = PosStopF
            cmd.motor_cmd[i].dq = VelStopF
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0

        while not self.stopped.wait(0.005):
            for i in range(12):
                cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
                cmd.motor_cmd[i].q = q0_real[i] + self.Δq_real[i]
                cmd.motor_cmd[i].dq = 0
                cmd.motor_cmd[i].kp = self.kp
                cmd.motor_cmd[i].kd = self.kd
                cmd.motor_cmd[i].tau = 0

            cmd.crc = self.crc.Crc(cmd)
            self.pub.Write(cmd)

    def get_obs(self):
        # shorthand
        motor_state_sim = [self.motor_state_real[i] for i in sim_idx_to_real_idx]

        joint_position = [ms.q - q0 for ms, q0 in zip(motor_state_sim, q0_sim)]
        joint_velocity = [ms.dq for ms in motor_state_sim]

        # imu.gyroscope & .rpy: std::array<float, 3>
        # imu.quaternion: std::array<float, 4>
        # => list[float]
        imu = self.imu
        gyroscope = imu.gyroscope  # rpy order, rad/s
        quaternion = imu.quaternion  # (w, x, y, z) order, normalized
        roll, pitch, yaw = imu.rpy  # rpy order, rad

        # rc = self.rc  # std::array<uint8_t, 40> => list[int]
        # stdlib struct.unpack is faster than convert tensor then .view(torch.float32)
        # and torch<=1.10 only support .view to dtype with same size
        # lx, rx, ry, _, ly = self.rocker_struct.unpack(bytes(rc[4:24]))
        # print(lx, rx, ry, ly)
        lx, ly, rx, ry = self.ball_vel

        # LSB -> MSB
        # rc[2] = [R1, L1, start, select][R2, L2, F1, F2]
        # rc[3] = [A, B, X, Y][up, right, down, left]
        # button_L1 = rc[2] & 0b0000_0010
        # button_L2 = rc[2] & 0b0010_0000
        # for byte in rc:
        #     print(f'{byte:08b}', end=' ')
        # print("button_L1: ", button_L1, "button_L2: ", button_L2)

        return RobotObservation(
            joint_position=joint_position,  # in sim order, relative to q0
            joint_velocity=joint_velocity,  # in sim order
            gyroscope=gyroscope,
            quaternion=quaternion,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            L1=self.L1,
            L2=self.L2,
        )

    def set_act(self, action: "list[float]"):
        # print(action)
        self.Δq_real = [action[i] for i in real_idx_to_sim_idx]

    def slowly_stand_up(self):
        import math

        stopped = self.stopped
        while any(math.isnan(Δq) for Δq in self.Δq_real) and not stopped.wait(0.05):
            pass

        Δq_sequence = []
        Δq_t = self.Δq_real
        while any(abs(Δq) > 0.01 for Δq in Δq_t):
            # reduce Δq magnitude by 0.05 rad per step
            Δq_t = [math.copysign(max(0, abs(Δq) - 0.05), Δq) for Δq in Δq_t]
            Δq_sequence.append(Δq_t)

        for Δq_t in Δq_sequence:
            self.Δq_real = Δq_t
            # 0.05 sec per step
            if stopped.wait(0.05):
                break

    def to_damp(self):
        # motorCmd.q = 0
        # motorCmd.dq = 0
        # motorCmd.Kp = 0
        # motorCmd.Kd = 2.0
        # motorCmd.tau = 0
        # self.Δq_real = [-q for q in q0_real]
        self.kp = 0.0
        self.kd = 10.0
        # self.q_setted = True

    def to_stand(self):
        # motorCmd.q = q_stand
        # motorCmd.dq = 0
        # motorCmd.Kp = kp * pd_ratio
        # motorCmd.Kd = kd * pd_ratio
        # motorCmd.tau = 0
        self.Δq_real = [0.0 for _ in range(12)]
        self.kp = 30.0
        self.kd = 5.0
        # self.kp = 20.0
        # self.kd = 0.5

    def to_run(self):
        self.kp = 20.0
        self.kd = 0.5

    def to_relax(self):
        self.kp = 0.0
        self.kd = 0.0

    def publish_ball_speed(self):
        msg = Float32MultiArray()
        msg.data = self.ball_vel[:2]
        self.publisher_.publish(msg)
