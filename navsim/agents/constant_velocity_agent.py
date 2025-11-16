"""
Constant velocity baseline agent implementation.

This module provides a simple baseline agent that predicts future trajectories
by assuming the ego vehicle will continue moving at its current velocity in a
straight line. No sensors or learning are required.
"""

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


class ConstantVelocityAgent(AbstractAgent):
    """
    Constant velocity baseline agent.

    This agent predicts trajectories by assuming the ego vehicle will continue
    moving forward at its current speed with no change in heading. This serves
    as a simple baseline for comparison with more sophisticated agents.
    """

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initialize the constant velocity agent.

        Args:
            trajectory_sampling: Configuration for trajectory sampling including
                time horizon and sampling interval. Defaults to 4 seconds at 0.5s intervals.
        """
        super().__init__(trajectory_sampling)

    def name(self) -> str:
        """
        Get the agent name.

        Returns:
            str: The class name of this agent.
        """
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Initialize the agent.

        This agent requires no initialization as it has no learnable parameters.
        """
        pass

    def get_sensor_config(self) -> SensorConfig:
        """
        Get the sensor configuration.

        Returns:
            SensorConfig: Configuration with no sensors, as this agent only uses
                ego vehicle velocity information.
        """
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Compute trajectory using constant velocity assumption.

        The predicted trajectory assumes the vehicle continues moving forward at
        its current speed with no steering or acceleration changes.

        Args:
            agent_input: Current state including ego vehicle velocity.

        Returns:
            Trajectory: Predicted future trajectory with poses at regular intervals.
        """
        # Get current ego vehicle velocity and compute speed magnitude
        ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5

        # Extract trajectory sampling parameters
        num_poses, dt = (
            self._trajectory_sampling.num_poses,
            self._trajectory_sampling.interval_length,
        )

        # Generate poses assuming constant forward motion at current speed
        # Format: [x_displacement, y_displacement, heading_change]
        poses = np.array(
            [[(time_idx + 1) * dt * ego_speed, 0.0, 0.0] for time_idx in range(num_poses)],
            dtype=np.float32,
        )

        return Trajectory(poses, self._trajectory_sampling)
