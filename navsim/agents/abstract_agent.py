"""
Abstract base class for autonomous driving agents in NAVSIM.

This module defines the interface that all agents must implement to work within the
NAVSIM framework. It provides the core methods for trajectory computation, feature
extraction, training, and sensor configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """
    Abstract base class for autonomous driving agents in NAVSIM.

    This class defines the interface that all agents must implement, including methods for:
    - Trajectory computation and prediction
    - Feature extraction from sensor data
    - Training loss computation and optimization
    - Sensor configuration specification

    Attributes:
        requires_scene: Whether the agent requires full scene information.
        _trajectory_sampling: Configuration for trajectory sampling parameters.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        requires_scene: bool = False,
    ):
        """
        Initialize the abstract agent.

        Args:
            trajectory_sampling: Configuration defining the trajectory sampling parameters
                (time horizon, sampling intervals, etc.).
            requires_scene: If True, the agent requires full scene information beyond
                immediate sensor data. Defaults to False.
        """
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling

    @abstractmethod
    def name(self) -> str:
        """
        Get the name identifier for this agent.

        Returns:
            str: A string describing the name of this agent implementation.
        """

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        Get the sensor configuration required by this agent.

        Returns:
            SensorConfig: Dataclass defining the sensor configuration including
                lidar parameters, camera specifications, and other sensor settings.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the agent and prepare it for inference or training.

        This method should perform any necessary setup operations such as loading
        pretrained weights, initializing internal states, or configuring components.
        """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a forward pass through the agent's neural network.

        Args:
            features: Dictionary mapping feature names to their corresponding tensors.
                The expected features depend on the specific agent implementation.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of predictions, typically including
                'trajectory' key with predicted ego vehicle trajectory.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        Get the list of feature builders for this agent.

        Feature builders extract and preprocess features from raw sensor data and
        scene information for input to the agent's neural network.

        Returns:
            List[AbstractFeatureBuilder]: List of feature builder instances.

        Raises:
            NotImplementedError: If the agent does not support training.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        Get the list of target builders for this agent.

        Target builders extract ground truth information from the scenario data
        for use in computing training losses.

        Returns:
            List[AbstractTargetBuilder]: List of target builder instances.

        Raises:
            NotImplementedError: If the agent does not support training.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Compute the predicted trajectory for the ego vehicle.

        This method orchestrates the full inference pipeline:
        1. Extract features from agent input using feature builders
        2. Run forward pass through the neural network
        3. Convert predictions to trajectory format

        Args:
            agent_input: Dataclass containing sensor data, map information,
                and other inputs needed by the agent.

        Returns:
            Trajectory: The predicted trajectory representing the ego vehicle's
                future positions and orientations.
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        # Build features from raw agent input using feature builders
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # Add batch dimension for neural network input
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # Run forward pass without gradient computation (inference mode)
        with torch.no_grad():
            predictions = self.forward(features)
            # Extract trajectory predictions and convert to numpy
            poses = predictions["trajectory"].squeeze(0).numpy()

        # Convert poses to Trajectory object with sampling configuration
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the training loss for backpropagation.

        Args:
            features: Dictionary of input features extracted from agent input.
            targets: Dictionary of ground truth targets extracted by target builders.
            predictions: Dictionary of model predictions from the forward pass.

        Returns:
            torch.Tensor: Scalar loss value used for backpropagation.

        Raises:
            NotImplementedError: If the agent does not support training.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        """
        Get the optimizer(s) and learning rate scheduler(s) for training.

        Returns:
            Either a single optimizer or a dictionary containing:
                - 'optimizer': The optimizer instance (e.g., Adam, SGD)
                - 'lr_scheduler': Optional learning rate scheduler

        Raises:
            NotImplementedError: If the agent does not support training.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Get PyTorch Lightning callbacks to be used during training.

        Callbacks can be used for logging, checkpointing, early stopping, etc.
        See navsim.planning.training.callbacks for implementation examples.

        Returns:
            List[pl.Callback]: List of PyTorch Lightning callback instances.
                Returns empty list by default if no callbacks are needed.
        """
        return []
