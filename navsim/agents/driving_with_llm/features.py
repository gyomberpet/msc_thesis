from typing import Any, Dict
import PIL.Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

class DrivingWithLLMFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "driving_with_llm_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Inherited, see superclass."""
        features = {}

        v = agent_input.ego_statuses[-1].ego_velocity
        a = agent_input.ego_statuses[-1].ego_acceleration
        driving_command = agent_input.ego_statuses[-1].driving_command

        features["status_feature"] = f"velocity: {v}, acceleration: {a}, driving_command: {driving_command}"
        features["camera_feature"] = self._get_camera_feature(agent_input)

        return features
    
    def _get_camera_feature(self, agent_input: AgentInput) -> list[PIL.Image.Image]:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return [transforms.ToPILImage()(tensor_image)]


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
