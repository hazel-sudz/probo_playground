import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


class Visualizer:
    """
    Visualizer for scenario simulation results.
    """

    def __init__(self, scenario_name: str):
        """
        Initialize the visualizer by loading CSV outputs and scenario config.

        Args:
            scenario_name: name of the scenario folder
        """
        base = Path(__file__).resolve().parent.parent
        self.output_path = base / "output" / scenario_name
        input_path = base / "input" / scenario_name

        self.gt_data = pd.read_csv(self.output_path / "ground_truth.csv")
        self.sensor_data = pd.read_csv(self.output_path / "sensor_data.csv")

        kf_path = self.output_path / "kalman_data.csv"
        self.kf_data = pd.read_csv(kf_path) if kf_path.exists() else None

        with open(input_path / "config.json", "r") as f:
            self.config = json.load(f)

    def plot_env(self):
        """
        Plot the environment features with no trajectories.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Environment Map")

        dims = self.config["dimensions"]
        ax.set_xlim(dims["x_min"] - 1, dims["x_max"] + 1)
        ax.set_ylim(dims["y_min"] - 1, dims["y_max"] + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        width = dims["x_max"] - dims["x_min"]
        height = dims["y_max"] - dims["y_min"]
        walls = patches.Rectangle(
            (dims["x_min"], dims["y_min"]),
            width,
            height,
            linewidth=5,
            edgecolor="black",
            facecolor="none",
            alpha=1.0,
        )
        ax.add_patch(walls)

        for obs in self.config["obstacles"]:
            width = obs["x_max"] - obs["x_min"]
            height = obs["y_max"] - obs["y_min"]
            rect = patches.Rectangle(
                (obs["x_min"], obs["y_min"]),
                width,
                height,
                linewidth=2,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
                label="Obstacle" if obs == self.config["obstacles"][0] else "",
            )
            ax.add_patch(rect)

        for i, lm in enumerate(self.config["landmarks"]):
            ax.plot(
                lm["x"],
                lm["y"],
                "r*",
                markersize=15,
                label="Landmark" if i == 0 else "",
            )
            ax.annotate(
                f"LM{lm['id']}",
                (lm["x"], lm["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
            )

        ax.legend(loc="upper right")
        return fig, ax

    def poses_from_gt(self):
        """
        Return ground truth poses as a DataFrame with columns: Time, x, y, theta.
        """
        return self.gt_data.rename(columns={
            "time": "Time",
            "robot_x": "x",
            "robot_y": "y",
            "robot_theta": "theta",
        })[["Time", "x", "y", "theta"]]

    def poses_from_kf(self):
        """
        Return KF predicted poses as a DataFrame with columns: Time, x, y, theta.
        Returns None if no KF data is available.
        """
        if self.kf_data is None:
            return None
        return self.kf_data.rename(columns={
            "t": "Time", "kf_x": "x", "kf_y": "y", "kf_theta": "theta",
        })[["Time", "x", "y", "theta"]]

    def poses_from_odom(self):
        """
        Integrate encoder velocities via dead reckoning to produce estimated poses.
        Returns a DataFrame with columns: Time, x, y, theta.
        """
        gt_start = self.gt_data.iloc[0]
        x = gt_start["robot_x"]
        y = gt_start["robot_y"]
        theta = gt_start["robot_theta"]
        dt = self.config["dt"]

        poses = []
        for _, row in self.sensor_data.iterrows():
            v = row.get("enc_lin_vel")
            w = row.get("enc_ang_vel")

            if pd.isna(v) or pd.isna(w):
                poses.append({"Time": row["time"], "x": x, "y": y, "theta": theta})
                continue

            x += np.cos(theta) * v * dt
            y += np.sin(theta) * v * dt
            theta += w * dt
            theta = theta % (2 * np.pi)

            poses.append({"Time": row["time"], "x": x, "y": y, "theta": theta})

        return pd.DataFrame(poses)

    def observed_landmarks(self):
        """
        Compute observed landmark positions from pinger bearing/range measurements
        using the ground truth robot pose at each timestep.

        Returns a DataFrame with columns: time, landmark_id, obs_x, obs_y.
        """
        lm_ids = [lm["id"] for lm in self.config["landmarks"]]
        observations = []

        for _, gt_row in self.gt_data.iterrows():
            t = gt_row["time"]
            rx = gt_row["robot_x"]
            ry = gt_row["robot_y"]
            rtheta = gt_row["robot_theta"]

            sensor_row = self.sensor_data[self.sensor_data["time"] == t]
            if sensor_row.empty:
                continue
            sensor_row = sensor_row.iloc[0]

            for lm_id in lm_ids:
                bearing_col = f"lm_{lm_id}_bearing"
                range_col = f"lm_{lm_id}_range"

                if bearing_col not in sensor_row.index or range_col not in sensor_row.index:
                    continue

                bearing = sensor_row[bearing_col]
                rng = sensor_row[range_col]

                if pd.isna(bearing) or pd.isna(rng) or not np.isfinite(rng):
                    continue

                obs_x = rx + rng * np.cos(rtheta + bearing)
                obs_y = ry + rng * np.sin(rtheta + bearing)

                observations.append({
                    "time": t,
                    "landmark_id": lm_id,
                    "obs_x": obs_x,
                    "obs_y": obs_y,
                })

        return pd.DataFrame(observations)

    def plot_single_trajectory(
        self,
        label,
        pose_table: pd.DataFrame,
        color="blue",
        alpha=1.0,
        scatter=False,
    ):
        """
        Plot a trajectory as a line with heading arrows on the current axes.
        """
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig, ax = self.plot_env()

        ax.plot(
            pose_table["x"],
            pose_table["y"],
            "-",
            color=color,
            linewidth=2,
            label=label,
            alpha=alpha,
        )

        skip = max(1, len(pose_table) // 20)
        for idx in range(0, len(pose_table), skip):
            row = pose_table.iloc[idx]
            dx = 0.5 * np.cos(row["theta"])
            dy = 0.5 * np.sin(row["theta"])
            ax.arrow(
                row["x"],
                row["y"],
                dx,
                dy,
                head_width=0.3,
                head_length=0.2,
                fc=color,
                ec=color,
                alpha=alpha * 0.25,
            )

        if scatter:
            ax.scatter(
                pose_table["x"],
                pose_table["y"],
                marker="*",
                color=color,
                linewidth=2,
                alpha=alpha,
            )

        start = pose_table.iloc[0]
        end = pose_table.iloc[-1]
        ax.plot(start["x"], start["y"], "o", color=color, markersize=10, alpha=alpha)
        ax.plot(end["x"], end["y"], "s", color=color, markersize=10, alpha=alpha)

        ax.legend(loc="upper right")
        return ax

    def draw_all(self):
        """
        Draw environment, ground truth trajectory, dead reckoning, and observed landmarks.
        """
        self.plot_env()
        self.plot_single_trajectory(
            "Ground Truth",
            self.poses_from_gt(),
            "green",
        )
        self.plot_single_trajectory(
            "Dead Reckoning",
            self.poses_from_odom(),
            "red",
        )

        kf_poses = self.poses_from_kf()
        if kf_poses is not None:
            self.plot_single_trajectory("KF Prediction", kf_poses, "blue")

        obs_lm = self.observed_landmarks()
        if not obs_lm.empty:
            ax = plt.gca()
            ax.scatter(
                obs_lm["obs_x"],
                obs_lm["obs_y"],
                c="orange",
                s=30,
                marker="x",
                alpha=0.4,
                label="Observed Landmarks",
                zorder=5,
            )
            ax.legend(loc="upper right")

        save_path = self.output_path / "trajectory.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    def animate_trajectories(
        self,
        fps=30,
        speedup=3.0,
        linger_seconds=5.0,
    ):
        """
        Create an animated GIF showing trajectories being drawn over time.
        """
        gt_poses = self.poses_from_gt()
        odom_poses = self.poses_from_odom()
        kf_poses = self.poses_from_kf()
        obs_lm = self.observed_landmarks()

        frame_counts = [len(gt_poses), len(odom_poses)]
        if kf_poses is not None:
            frame_counts.append(len(kf_poses))
        max_frames = max(frame_counts)
        frame_skip = int(speedup)
        frame_indices = list(range(0, max_frames, max(1, frame_skip)))

        linger_frames = int(linger_seconds * fps)
        frame_indices.extend([frame_indices[-1]] * linger_frames)

        fig, ax = self.plot_env()

        (gt_line,) = ax.plot(
            [], [], "-", color="green", linewidth=2, label="Ground Truth", alpha=0.8
        )
        (odom_line,) = ax.plot(
            [], [], "-", color="red", linewidth=2, label="Dead Reckoning", alpha=0.8
        )
        (kf_line,) = ax.plot(
            [], [], "-", color="blue", linewidth=2, label="KF Prediction", alpha=0.8
        ) if kf_poses is not None else (None,)
        lm_scatter = ax.scatter(
            [],
            [],
            c="orange",
            s=30,
            marker="x",
            label="Observed Landmarks",
            alpha=0.4,
            zorder=5,
        )

        gt_end = ax.plot([], [], "s", color="green", markersize=10, alpha=0)[0]
        odom_end = ax.plot([], [], "s", color="red", markersize=10, alpha=0)[0]
        kf_end = ax.plot([], [], "s", color="blue", markersize=10, alpha=0)[0] if kf_poses is not None else None

        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.legend(loc="upper right")

        def init():
            gt_line.set_data([], [])
            odom_line.set_data([], [])
            if kf_line is not None:
                kf_line.set_data([], [])
            lm_scatter.set_offsets(np.empty((0, 2)))
            gt_end.set_data([], [])
            odom_end.set_data([], [])
            if kf_end is not None:
                kf_end.set_data([], [])
            time_text.set_text("")
            artists = [gt_line, odom_line, lm_scatter, gt_end, odom_end, time_text]
            if kf_line is not None:
                artists.insert(2, kf_line)
            if kf_end is not None:
                artists.insert(-1, kf_end)
            return tuple(artists)

        def animate(frame_idx):
            actual_frame = (
                frame_indices[frame_idx]
                if frame_idx < len(frame_indices)
                else frame_indices[-1]
            )
            is_final_frame = frame_idx >= len(frame_indices) - linger_frames

            if actual_frame < len(gt_poses):
                gt_data = gt_poses.iloc[: actual_frame + 1]
                gt_line.set_data(gt_data["x"], gt_data["y"])
                current_time = gt_data.iloc[-1]["Time"]
                time_text.set_text(f"Time: {current_time:.1f}s")

                if is_final_frame:
                    gt_end.set_data([gt_data.iloc[-1]["x"]], [gt_data.iloc[-1]["y"]])
                    gt_end.set_alpha(0.8)

            if actual_frame < len(odom_poses):
                odom_data = odom_poses.iloc[: actual_frame + 1]
                odom_line.set_data(odom_data["x"], odom_data["y"])

                if is_final_frame:
                    odom_end.set_data(
                        [odom_data.iloc[-1]["x"]], [odom_data.iloc[-1]["y"]]
                    )
                    odom_end.set_alpha(0.8)

            if kf_poses is not None and actual_frame < len(kf_poses):
                kf_data = kf_poses.iloc[: actual_frame + 1]
                kf_line.set_data(kf_data["x"], kf_data["y"])

                if is_final_frame:
                    kf_end.set_data(
                        [kf_data.iloc[-1]["x"]], [kf_data.iloc[-1]["y"]]
                    )
                    kf_end.set_alpha(0.8)

            if actual_frame < len(gt_poses) and not obs_lm.empty:
                current_time = gt_poses.iloc[actual_frame]["Time"]
                obs_up_to_now = obs_lm[obs_lm["time"] <= current_time]
                if len(obs_up_to_now) > 0:
                    lm_scatter.set_offsets(obs_up_to_now[["obs_x", "obs_y"]].values)

            artists = [gt_line, odom_line, lm_scatter, gt_end, odom_end, time_text]
            if kf_line is not None:
                artists.insert(2, kf_line)
            if kf_end is not None:
                artists.insert(-1, kf_end)
            return tuple(artists)

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frame_indices),
            interval=1000 / fps,
            blit=True,
            repeat=True,
        )

        writer = PillowWriter(fps=fps)
        save_path = self.output_path / "trajectory_animation.gif"
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"Saved animation to {save_path}")
        return anim


if __name__ == "__main__":
    scenario = sys.argv[1]
    vis = Visualizer(scenario)
    vis.draw_all()
    vis.animate_trajectories()
