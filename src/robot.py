"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

from environment import Environment
from sensors import SensorInterface
from typing import List
import math
import pandas as pd

class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        self.env = env
        self.sensors: List[SensorInterface] = []
    
    def _angle_norm(self, angle: float):
        """
            Normalize an angle with ring-mod
        """
        return ((angle % (2*math.pi)) + (2*math.pi)) % (2*math.pi)

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """

        l = lin_vel * self.env.DT
        dx = math.cos(self.env.robot_pose.theta) * l
        dy = math.sin(self.env.robot_pose.theta) * l
        dtheta = ang_vel * self.env.DT

        self.env.robot_step(dx, dy, dtheta)

        return (dx, dy, dtheta)

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT
        dtheta = ang_vel * self.env.DT

        self.env.robot_step(dx, dy, dtheta)

        return (dx, dy, dtheta)

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        row = {"time": self.env.time}
        for sensor in self.sensors:
            dt = self.env.time - sensor.last_meas_t
            if dt >= sensor.interval - 1e-9:
                reading = sensor.sample()
                if sensor.name == "wheel_encoder":
                    row["enc_x_vel"] = reading.x
                    row["enc_y_vel"] = reading.y
                    row["enc_ang_vel"] = reading.angular
                elif sensor.name == "landmark_pinger":
                    for br in reading:
                        row[f"lm_{int(br.landmark_id)}_bearing"] = br.bearing
                        row[f"lm_{int(br.landmark_id)}_range"] = br.range
                elif sensor.name == "gps": 
                    row["gps_x"] = reading[0]
                    row["gps_y"] = reading[1]
        return pd.DataFrame([row])
