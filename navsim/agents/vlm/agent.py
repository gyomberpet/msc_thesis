from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from torch.cuda.amp import autocast
from transformers import AutoProcessor, AutoModelForImageTextToText
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import random

class VisionLanguageModel():
    def __init__(self):
        self.model_name="Qwen/Qwen2-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, torch_dtype=torch.float16)

        self.prompt = """You receive an image from the driver's perspective. Your task is to describe the image in high details,
        so the driver can make informed driving decisions. Only focus on objects related to driving, like vehicles, pedestrian traffic lights etc."""
        self.max_tokens = 500
        
    def describe_image(self, image_path) -> str:
        messages = [
            { 
                "role": "system", 
                "content": "You are an Autonomous Driving AI assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_path}",
                        "resized_height": 256,
                        "resized_width": 1024,
                    },
                    {"type": "text", "text": f"{self.prompt}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with autocast():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    
class LargeLanguageModel():
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_tokens=512
        self.trajectory_sampling = trajectory_sampling

    def forward(self, traffic_description: str) -> str:
        example = [[random.uniform(0, 1) for _ in range(3)] for _ in range(10)]
        prompt = f"""Your task is to predict the optimal future waypoints of the vehicle, that you are driving, based on the description of current traffic conditions.
        Here is the description: {traffic_description}
        Your response should only be a list with exaclty {self.trajectory_sampling.num_poses} items, where each item is a 3 length long list, representing a single waypoint of float numbers (x, y, heading).
        Example format of response: {example}
        You have to return a {self.trajectory_sampling.num_poses}*3 flaot list even if you are not sure about the exact numbers. Please write the numbers, that you think alligns the best with the description."""
        messages = [
            {"role": "system", "content": "You are an Autonomous Driving AI assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

class VLMAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._vlm = VisionLanguageModel()
        self._transfuser_model = TransfuserModel(config)

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
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
