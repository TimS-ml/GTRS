"""
Human agent implementation using ground truth future trajectory.

This module provides a privileged agent that has access to the actual future
trajectory from the logged data. It serves as an oracle/upper-bound baseline
for evaluating autonomous agents.
"""

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class HumanAgent(AbstractAgent):
    """
    Privileged agent that returns the ground truth human-driven trajectory.

    This agent has access to the actual future trajectory from logged data,
    making it an oracle that represents perfect prediction. It is used as
    an upper-bound baseline for comparison with autonomous agents.

    Note: This agent requires scene information and cannot be used in
    real-world deployment scenarios.
    """

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initialize the human agent.

        Args:
            trajectory_sampling: Configuration for trajectory sampling including
                time horizon and sampling interval. Defaults to 4 seconds at 0.5s intervals.
        """
        self._trajectory_sampling = trajectory_sampling

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

        This agent requires no initialization as it directly returns ground truth.
        """
        pass

    def get_sensor_config(self) -> SensorConfig:
        """
        Get the sensor configuration.

        Returns:
            SensorConfig: Configuration with no sensors, as this agent uses
                privileged ground truth information instead of sensor data.
        """
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Return the ground truth future trajectory from the scene.

        This method extracts the actual human-driven trajectory from the logged
        scene data, providing perfect prediction for evaluation purposes.

        Args:
            agent_input: Current agent inputs (not used, as agent uses scene directly).
            scene: Scene containing the ground truth future trajectory.

        Returns:
            Trajectory: The actual future trajectory that was executed in the logged data.
        """
        return scene.get_future_trajectory(self._trajectory_sampling.num_poses)
