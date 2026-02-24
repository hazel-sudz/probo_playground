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
                "linear_vel": float(row["linear_vel"]),
                "angular_vel": float(row["angular_vel"]),
            })

    # Sim loop
    total_timesteps = int(total_seconds / env.DT)
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = ""
    output_ground_truth_filepath = ""
    output_sensor_data_filepath = ""

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # TODO: take a ground truth snapshot and add it to the history

            # TODO: take sensor measurements and add it to the history

            # TODO: retrieve the next motor command from the input file

            # TODO: execute the motor command

    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file

    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
