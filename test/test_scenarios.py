from pathlib import Path

import pytest
from main import run_scenario
from vis import Visualizer


@pytest.fixture(scope="class")
def movement_only_data(request):
    gt, sensor, kalman = run_scenario("movement_only")

    output_dir = Path(__file__).resolve().parent.parent / "output" / "movement_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output_dir / "ground_truth.csv", index=False)
    sensor.to_csv(output_dir / "sensor_data.csv", index=False)
    kalman.to_csv(output_dir / "kalman_data.csv", index=False)

    vis = Visualizer("movement_only")
    vis.draw_all()
    vis.animate_trajectories()

    request.cls.gt = gt
    request.cls.sensor = sensor
    request.cls.kalman = kalman


@pytest.fixture(scope="class")
def rotation_only_data(request):
    gt, sensor, kalman = run_scenario("rotation_only")

    output_dir = Path(__file__).resolve().parent.parent / "output" / "rotation_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output_dir / "ground_truth.csv", index=False)
    sensor.to_csv(output_dir / "sensor_data.csv", index=False)
    kalman.to_csv(output_dir / "kalman_data.csv", index=False)

    vis = Visualizer("rotation_only")
    vis.draw_all()
    vis.animate_trajectories()

    request.cls.gt = gt
    request.cls.sensor = sensor
    request.cls.kalman = kalman


@pytest.mark.usefixtures("movement_only_data")
class TestMovementOnly:
    def test_x_increases(self):
        """Robot x position should increase over time (moving forward at heading=0)."""
        assert self.gt["robot_x"].iloc[-1] > self.gt["robot_x"].iloc[0]

    def test_y_stays_near_zero(self):
        """Robot y position should stay near 0 (heading=0, no angular vel)."""
        assert self.gt["robot_y"].abs().max() < 1e-6

    def test_theta_stays_zero(self):
        """Robot theta should remain 0 throughout."""
        assert (self.gt["robot_theta"] == 0.0).all()

    def test_enc_lin_vel_present_and_nonzero(self):
        """Encoder linear velocity columns should be present and non-zero when sampled."""
        assert "enc_lin_vel" in self.sensor.columns
        sampled = self.sensor["enc_lin_vel"].dropna()
        assert len(sampled) > 0
        assert sampled.abs().sum() > 0

    def test_enc_ang_vel_near_zero(self):
        """Encoder angular velocity should be near zero."""
        assert "enc_ang_vel" in self.sensor.columns
        sampled = self.sensor["enc_ang_vel"].dropna()
        assert sampled.abs().mean() < 0.5


@pytest.mark.usefixtures("rotation_only_data")
class TestRotationOnly:
    def test_x_stays_zero(self):
        """Robot x position should stay at 0 (no linear vel)."""
        assert self.gt["robot_x"].abs().max() < 1e-6

    def test_y_stays_zero(self):
        """Robot y position should stay at 0."""
        assert self.gt["robot_y"].abs().max() < 1e-6

    def test_theta_changes(self):
        """Robot theta should change over time."""
        assert self.gt["robot_theta"].nunique() > 1

    def test_enc_lin_vel_near_zero(self):
        """Encoder linear velocity should be near zero."""
        assert "enc_lin_vel" in self.sensor.columns
        sampled = self.sensor["enc_lin_vel"].dropna()
        assert sampled.abs().mean() < 0.5

    def test_enc_ang_vel_nonzero(self):
        """Encoder angular velocity should be non-zero when sampled."""
        assert "enc_ang_vel" in self.sensor.columns
        sampled = self.sensor["enc_ang_vel"].dropna()
        assert len(sampled) > 0
        assert sampled.abs().sum() > 0
