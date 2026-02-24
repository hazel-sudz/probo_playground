"""
Main file for running the simulator.
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from environment import Environment
from robot import Robot
from sensors import WheelEncoder, LandmarkPinger, GPS
from extended_kalman_filter import ExtendedKalmanFilter
from kalman_filter import KalmanFilter
from utils import Position, Pose, Landmark, Bounds


def run_scenario(scenario_name):
    """
    Run a simulation scenario by loading config and velocity commands from
    input/{scenario_name}/, executing the sim loop, and returning results.

    Args:
        scenario_name: name of the scenario folder under input/

    Returns:
        (ground_truth_df, sensor_data_df): concatenated DataFrames of all
        ground truth snapshots and sensor measurements.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "input", scenario_name)

    # Load config
    with open(os.path.join(base_dir, "config.json"), "r") as f:
        config = json.load(f)

    dims = config["dimensions"]
    dimensions = Bounds(dims["x_min"], dims["x_max"], dims["y_min"], dims["y_max"])
    dt = config["dt"]
    total_seconds = config["total_seconds"]

    obstacles = []
    for obs in config["obstacles"]:
        obstacles.append(Bounds(obs["x_min"], obs["x_max"], obs["y_min"], obs["y_max"]))

    landmarks = []
    for lm in config["landmarks"]:
        landmarks.append(Landmark(pos=Position(lm["x"], lm["y"]), id=lm["id"]))

    rp = config["initial_robot_pose"]
    initial_robot_pose = Pose(pos=Position(rp["x"], rp["y"]), theta=rp["theta"])

    # Create environment and robot
    env = Environment(dimensions, dt, obstacles, landmarks, initial_robot_pose)
    robot = Robot(env)

    # set up the (Extended) Kalman Filter
    LINEAR = True
    if LINEAR:
        kf = KalmanFilter(
            dt,
            np.array([initial_robot_pose.pos.x, initial_robot_pose.pos.y, initial_robot_pose.theta]),
        )
    else:
        # set up the Extended Kalman Filter
        kf = ExtendedKalmanFilter(
            dt,
            np.array([initial_robot_pose.pos.x, initial_robot_pose.pos.y, initial_robot_pose.theta]),
        )

    # Attach sensors
    encoder = WheelEncoder(robot)
    pinger = LandmarkPinger(robot)
    gps = GPS(robot, name="gps", interval=1.0, x_noise=0.5, y_noise=0.5)
    robot.sensors.append(encoder)
    robot.sensors.append(pinger)
    robot.sensors.append(gps)

    # Load velocity commands
    vel_commands = []
    with open(os.path.join(base_dir, "vel_cmd.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vel_commands.append({
                "timestamp": float(row["timestamp"]),
                "x_vel": float(row["x_vel"]),
                "y_vel": float(row["y_vel"]),
                "angular_vel": float(row["angular_vel"]),
            })

    # Sim loop
    total_timesteps = int(total_seconds / env.DT)
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    cmd_index = 0
    u_x = 0.0
    u_y = 0.0
    u_theta = 0.0

    for step in range(total_timesteps + 1):
        # Snapshot ground truth
        ground_truth_history.append(env.take_state_snapshot())

        # Take sensor measurements
        sensor_data_history.append(robot.take_sensor_measurements())

        if LINEAR:
            x, p = kf.predict(np.array([u_x, u_y, u_theta]))
            kalman_filter_history.append({
                "t": env.time,
                "kf_x": float(x[0]),
                "kf_y": float(x[1]),
                "kf_theta": float(x[2]),
                "p_xx": float(p[0, 0]),
                "p_yy": float(p[1, 1]),
                "p_tt": float(p[2, 2]),
            })
            # TODO: call the Kalman Filter update step if new sensor data is available
            pass
        else:
            # TODO: call the Extended Kalman Filter prediction step

            # TODO: call the Extended Kalman Filter update step if new sensor data is available, for each GPS reading and for each landmark ping
            pass

        # Check if a new command should be applied at this timestamp
        while cmd_index < len(vel_commands) and vel_commands[cmd_index]["timestamp"] <= env.time:
            u_x = vel_commands[cmd_index]["x_vel"]
            u_y = vel_commands[cmd_index]["y_vel"]
            u_theta = vel_commands[cmd_index]["angular_vel"]
            cmd_index += 1

        # Execute motor command
        robot.robot_step_translational(u_x, u_y, u_theta)

        # Advance time
        env.time += env.DT

    ground_truth_df = pd.concat(ground_truth_history, ignore_index=True)
    sensor_data_df = pd.concat(sensor_data_history, ignore_index=True)
    kalman_data_df = pd.DataFrame(kalman_filter_history)

    return ground_truth_df, sensor_data_df, kalman_data_df


if __name__ == "__main__":
    scenario = sys.argv[1]
    gt_df, sensor_df, kalman_df = run_scenario(scenario)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", scenario)
    os.makedirs(output_dir, exist_ok=True)

    gt_df.to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)
    sensor_df.to_csv(os.path.join(output_dir, "sensor_data.csv"), index=False)
    kalman_df.to_csv(os.path.join(output_dir, "kalman_data.csv"), index=False)
    print(f"Saved outputs to {output_dir}")
