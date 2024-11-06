from typing import Any, List, Dict, Optional, Union
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory, AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.mycustom.mycustom_features import MyCustomFeatureBuilder, MyCustomTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from hydra.utils import instantiate


class MyCustomAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes the agent interface for MyCustomAgent.
        :param trajectory_sampling: trajectory sampling specification.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )

    def name(self) -> str:
        return self.__class__.__name__
    
    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])
    
    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [MyCustomTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [MyCustomFeatureBuilder()]
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poses: torch.Tensor = self._model(features["status_feature"])
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}
    
    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])
    
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._model.parameters(), lr=self._lr)

agent = instantiate("ego_status_mlp_agent")
print(agent.name())