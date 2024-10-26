# dribble deploy go2

> [!IMPORTANT]  
> Work in process

## Usage Instructions

### Start and stop running

Modify model path at deploy.py#L16, turn off the motion service, make dog prone, then:

```bash
export ROS_DOMAIN_ID=1
python3 fisheye_pub_ball_location.py
```

```bash
export ROS_DOMAIN_ID=1
python3 deploy.py
```

Press L1 on the remote control, the robot will stand up, press L1 again the dribblebot model will start inference and take control. At this time, use the left joystick to control the direction of the ball. Press L2 to stop, and the robot will slowly fall back to the prone state.

### Visualize ball detection

```bash
export ROS_DOMAIN_ID=1
python3 visualize.py
```

### References

- [Dribblebot](https://github.com/Improbable-AI/dribblebot)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
