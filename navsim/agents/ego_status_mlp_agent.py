"""
Simple MLP-based agent using ego vehicle status for trajectory prediction.

This module implements a baseline learning-based agent that uses a multi-layer
perceptron to predict trajectories based solely on the ego vehicle's current
velocity, acceleration, and driving command.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder that extracts ego vehicle status information.

    Extracts and concatenates velocity, acceleration, and driving command
    into a single feature vector for input to the EgoStatusMLP agent.
    """

    def __init__(self):
        """Initialize the ego status feature builder."""
        pass

    def get_unique_name(self) -> str:
        """
        Get the unique name for this feature builder.

        Returns:
            str: Unique identifier for ego status features.
        """
        return "ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Extract ego status features from agent input.

        Concatenates velocity (2D), acceleration (2D), and driving command (4D)
        into an 8-dimensional feature vector.

        Args:
            agent_input: Agent input containing ego vehicle status history.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'ego_status' key containing
                the concatenated 8D feature vector.
        """
        # Get the most recent ego status
        ego_status = agent_input.ego_statuses[-1]

        # Extract individual components
        velocity = torch.tensor(ego_status.ego_velocity)
        acceleration = torch.tensor(ego_status.ego_acceleration)
        driving_command = torch.tensor(ego_status.driving_command)

        # Concatenate into single feature vector
        ego_status_feature = torch.cat([velocity, acceleration, driving_command], dim=-1)
        return {"ego_status": ego_status_feature}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """
    Target builder that extracts ground truth trajectory.

    Retrieves the actual future trajectory from the scene for computing
    training loss against the agent's predictions.
    """

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initialize the trajectory target builder.

        Args:
            trajectory_sampling: Specification for trajectory sampling including
                number of poses and time intervals.
        """
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """
        Get the unique name for this target builder.

        Returns:
            str: Unique identifier for trajectory targets.
        """
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Extract ground truth trajectory from scene.

        Args:
            scene: Scene containing the ground truth future trajectory.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'trajectory' key containing
                tensor of shape (num_poses, 3) with [x, y, heading] for each pose.
        """
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class EgoStatusMLPAgent(AbstractAgent):
    """
    Simple MLP-based agent using only ego vehicle status.

    This agent uses a 3-hidden-layer MLP to predict future trajectories based
    solely on the current ego vehicle status (velocity, acceleration, driving
    command). It serves as a baseline to evaluate the benefit of additional
    sensor information.

    Attributes:
        _checkpoint_path: Path to pretrained model weights.
        _lr: Learning rate for optimizer.
        _mlp: The multi-layer perceptron neural network.
    """

    def __init__(
        self,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initialize the EgoStatusMLP agent.

        Args:
            hidden_layer_dim: Dimensionality of hidden layers in the MLP.
            lr: Learning rate for training the agent.
            checkpoint_path: Optional path to pretrained model checkpoint for loading
                weights. Defaults to None.
            trajectory_sampling: Configuration for trajectory sampling. Defaults to
                4 second horizon at 0.5s intervals.
        """
        super().__init__(trajectory_sampling)

        self._checkpoint_path = checkpoint_path
        self._lr = lr

        # Build MLP: 8 inputs -> hidden layers -> trajectory outputs
        # Input: [velocity(2), acceleration(2), driving_command(4)] = 8 dimensions
        # Output: num_poses * 3 (x, y, heading for each future pose)
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )

    def name(self) -> str:
        """
        Get the agent name.

        Returns:
            str: The class name of this agent.
        """
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Initialize the agent by loading pretrained weights.

        Loads model weights from the checkpoint path. Automatically handles
        GPU/CPU device mapping and removes 'agent.' prefix from state dict keys.
        """
        # Load checkpoint with appropriate device mapping
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        # Remove 'agent.' prefix from keys to match model structure
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """
        Get the sensor configuration.

        Returns:
            SensorConfig: Configuration with no sensors, as this agent only
                uses ego vehicle status information.
        """
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        Get the list of target builders.

        Returns:
            List[AbstractTargetBuilder]: List containing the trajectory target builder.
        """
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        Get the list of feature builders.

        Returns:
            List[AbstractFeatureBuilder]: List containing the ego status feature builder.
        """
        return [EgoStatusFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MLP.

        Args:
            features: Dictionary containing 'ego_status' tensor of shape (batch, 8).

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'trajectory' key containing
                predictions of shape (batch, num_poses, 3).
        """
        # Pass ego status through MLP and reshape to trajectory format
        poses: torch.Tensor = self._mlp(features["ego_status"].to(torch.float32))
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute L1 loss between predicted and ground truth trajectories.

        Args:
            features: Dictionary of input features (not used in loss computation).
            targets: Dictionary containing ground truth 'trajectory' tensor.
            predictions: Dictionary containing predicted 'trajectory' tensor.

        Returns:
            torch.Tensor: Scalar L1 loss value.
        """
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Get the optimizer for training.

        Returns:
            Optimizer: Adam optimizer with configured learning rate.
        """
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)
