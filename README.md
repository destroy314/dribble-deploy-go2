# dribble deploy go2

> [!IMPORTANT]  
> Work in progress
 
TODO: 
1. Control freq. seems too slow.
2. Remove ROS component.

## Usage Instructions

### Close sport mode

```bash
cd scripts
python3 close_sport_mode.py
```

### Start and stop running

Modify model path at `deploy.py`: load_policy(), make dog prone, then:

```bash
export ROS_DOMAIN_ID=1
python3 fisheye_pub_ball_location.py
```

```bash
export ROS_DOMAIN_ID=1
python3 deploy.py
```

Press L1 on the remote controller, the robot will stand up, press L1 again the dribblebot model will start inference and take control. At this time, use the left joystick to control the direction of the ball. Press L2 to stop, and the robot will slowly fall back to the prone state.

### Visualize ball detection

```bash
export ROS_DOMAIN_ID=1
python3 scripts/visualize.py
```

### Observations definition

1. ball_pos: ball position in base frame (ObjectSensor)
    d = 3
    ```
    v = (R_base^-1 @ (p_ball - p_base)) * [1, 1, 0]
    ```
    scale = 1.0
2. projected_gravity (OrientationSensor)
    d = 3
    ```
    v = R_base^-1 @ [0, 0, -1]
    ```
    scale = N/A # 1.0
3. commands (RCSensor)
    d = 15
    ```
    v = [
        0: x_vel = lin_vel[0],
        1: y_vel = lin_vel[1],
        2: yaw_vel = 0, # train = 0
        3: body_height = 0, # train in [-0.05, 0.05]
        4: step_frequency = 3, # train = 3
        5..8: gait = [phase, offset, bound] = trotting = [0.5, 0, 0], # train = [0.5, 0, 0]
        8: duration = 0.5, # train = 0.5 # duration = stance / (stance + swing)
        9: footswing_height = 0.09, # train = 0.09
        10: pitch = 0, # train = 0
        11: roll = 0, # train = 0
        12: stance_width = 0, # train in [0, 0.1]
        13: stance_length = uniform(0, 0.1), # train in [0, 0.1]
        14: unknown (aux_reward_coef) = uniform(0, 0.01), # train in  [0, 0.01]
    ]
    ```
    ```
    scale = [
        lin_vel = 2, lin_vel, ang_vel = 0.25,
        body_height_cmd = 2, gait_freq_cmd = 1,
        gait_phase = 1, gait_phase, gait_phase, gait_phase,
        footswing_height = 0.15, pitch = 0.3, roll = 0.3,
        stance_width = 1, stance_length = 1,
        aux_reward = 1,
    ]
    ```
4. dof_pos (JointPositionSensor)
    d = 12
    dof_pos is relative to default pos, i.e. `motorState.q - q0`.
    ```
    q0 = {
        'FL_hip_joint': 0.1,
        'RL_hip_joint': 0.1,
        'FR_hip_joint': -0.1,
        'RR_hip_joint': -0.1,
        'FL_thigh_joint': 0.8,
        'RL_thigh_joint': 1.0,
        'FR_thigh_joint': 0.8,
        'RR_thigh_joint': 1.0,
        'FL_calf_joint': -1.5,
        'RL_calf_joint': -1.5,
        'FR_calf_joint': -1.5,
        'RR_calf_joint': -1.5,
    }
    ```
    `print(env.dof_names)` output
    ```
    ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    ```
    This index-name mapping is used to computed default pos.
    dof_pos should be consistent with it.
    scale = 1
5. dof_vel (JointVelocitySensor)
    d = 12
    scale = 0.05
6. action: a_{t-1} (ActionSensor)
    d = 12
    scale = N/A # 1.0
    Note: 0 init
7. last_action: a_{t-2} (ActionSensor(delay=1) / LastActionSensor(delay=1))
    d = 12
    scale = N/A # 1.0
    Note: 0 init
8. clock (ClockSensor)
    d = 4
    ```
    dt = control.decimation * sim.dt = 4 * 0.005 = 0.02 # 50Hz
    per step: gait_index = (gait_index + dt * step_frequency) % 1 # scalar
    foot_indices = (gait_index + [phase + offset + bound, offset, bound, phase]) # pacing_offset = False
    v = sin(2π foot_indices)
    ```
    scale = N/A # 1.0
9. yaw (YawSensor)
    d = 1
    ```
    f = R_base @ [1, 0, 0]
    v = atan2(f[1], f[0]) # wrap to [-π, π]
    ```
    scale = N/A # 1.0
10. timing (TimingSensor)
    d = 1
    ```
    v = gait_index # see clock
    ```
    scale  = N/A # 1.0

### References

- [Dribblebot](https://github.com/Improbable-AI/dribblebot)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)