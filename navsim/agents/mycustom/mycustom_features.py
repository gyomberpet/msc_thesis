from typing import Dict
import torch
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

class MyCustomFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self):
        pass

    def get_unique_name(self) -> str:
        return "mycustom_feature"
    
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {}

        # features["camera_feature"] = self._get_amera_feature(agent_input)
        # features["lidar_feature"] = self._get_lcidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features
    
class MyCustomTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "mycustom_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        targets = {}
        
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        targets["trajectory"] = torch.tensor(future_trajectory.poses)

        return targets