import torch
from typing import Any, List, Dict, Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.driving_with_llm.models import LoRAWithMLP, VLM
from transformers import AutoTokenizer
from navsim.agents.driving_with_llm.features import TrajectoryTargetBuilder, DrivingWithLLMFeatureBuilder

class DrivingWithLLMAgent(AbstractAgent):
    """DrivingWithLLMAgent agent interface."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
        vlm_name = "Qwen/Qwen2-VL-7B-Instruct"

        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._vlm = VLM(base_model_name=vlm_name)
        self._llm_lora = LoRAWithMLP(
            base_model_name=llm_name,
            trajectory_sampling=trajectory_sampling,
            hidden_dim=hidden_layer_dim
        )
        self._tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def _tokenize_function(self, data):
        text = f"""Your task is to decide whether to slow down (1) or continue (0) based on traffic conditions.
        Here is the description: {data['text']}
        You must return only a single integer: 0 or 1."""
        
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}  # Remove batch dim
        return tokens


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(True)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [DrivingWithLLMFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        with torch.no_grad():
            image_descriptions: list[str] = self._vlm(features["camera_feature"])
            #image_descriptions[0] += f"\n\n Ego status\n {features["status_feature"]}" 
            tokens = self._tokenize_function(image_descriptions)

        poses: torch.Tensor = self._llm_lora(tokens["input_ids"])
        
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)
