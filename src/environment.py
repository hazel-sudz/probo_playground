"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

from utils import Position, Pose, Bounds, Landmark, BearingRange
import math
import pandas as pd


class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        self.DIMENSIONS = dimensions

        self.DT = dt

        self.time = 0.0

        self.OBSTACLES = obstacles 
        self.LANDMARKS = landmarks

        self.robot_pose = robot_starting_pose

    def robot_step(self, dx: float, dy: float, dtheta: float):
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        self.robot_pose.pos = self.is_valid_motion(dx, dy)
        self.robot_pose.theta = (self.robot_pose.theta + dtheta) % 360

    def is_valid_motion(self, dx: float, dy: float) -> Position:
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """

        current_pose = self.robot_pose.deep_copy()
        next_pose = current_pose.deep_copy()

        next_pose.pos.x += dx
        next_pose.pos.y += dy

        if self.is_valid_position(next_pose):
            return next_pose.pos
        return current_pose.pos

    def is_valid_position(self, position: Position):
        """
        Check if a given robot position is valid; i.e. not out-of-bounds or within an obstacle. Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """

        # check map boundary
        if not self.DIMENSIONS.within_bounds(position):
            return False

        # check obstacles
        for bound in self.OBSTACLES:
            if not bound.within_bounds(position):
                return False

        return True 

    def get_robot_pose(self):
        """
        Return the true robot pose.
        """
        # TODO: fill in the function
        pass
    

    def _get_proximity_to_landmark(self, lm: Landmark):
        """
        Return the robot's range and bearing to a given landmark
        """

        dx = lm.pos.x - self.robot_pose.pos.x
        dy = lm.pos.y - self.robot_pose.pos.y
        range = math.sqrt(dx**2 + dy**2)

        # angle from x-axis (radians)
        dtheta = math.atan2(dy, dx)

        # offset angle by current bearing (convert theta from degrees to radians)
        dtheta -= math.radians(self.robot_pose.theta)

        # norm with ring-mod
        dtheta = (dtheta + 2*math.pi) % (2*math.pi)

        return BearingRange(
            landmark_id=lm.id,
            bearing=dtheta,
            range=range
        )

    def get_proximity_to_landmarks(self):
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        return [self._get_proximity_to_landmark(lm) for lm in self.LANDMARKS]

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        proximity = self.get_proximity_to_landmarks()
        row = {
            "time": self.time,
            "robot_x": self.robot_pose.pos.x,
            "robot_y": self.robot_pose.pos.y,
            "robot_theta": self.robot_pose.theta,
        }
        for br in proximity:
            row[f"lm_{int(br.landmark_id)}_bearing"] = br.bearing
            row[f"lm_{int(br.landmark_id)}_range"] = br.range
        return pd.DataFrame([row])

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        return {
            "dimensions": self.DIMENSIONS.to_dict(),
            "dt": self.DT,
            "obstacles": [obs.to_dict() for obs in self.OBSTACLES],
            "landmarks": [lm.to_dict() for lm in self.LANDMARKS],
        }
