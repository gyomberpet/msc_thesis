from typing import Any, List, Dict, Optional, Union

from navsim.agents.llm_lora.transfuser_backbone import TransfuserBackbone
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.llm_lora.llm_config import LLMConfig
from navsim.agents.llm_lora.transfuser_model import TransfuserModel
from navsim.agents.llm_lora.transfuser_callback import TransfuserCallback
from navsim.agents.llm_lora.transfuser_loss import transfuser_loss
from navsim.agents.llm_lora.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from driving_with_llms.utils.model_utils import load_model, load_llama_tokenizer
from driving_with_llms.utils.training_utils import tokenize, decode_generation_seqeunces
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig, LlamaForCausalLM
import torch.nn.functional as F

BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"

class SimplifiedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.weighted_loss_on_numbers = True
        if self.weighted_loss_on_numbers:
            number_tokens = [
                448, 29900, 29889, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929
            ]
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0  # Higher weight on certain tokens (e.g., numbers)
            self.register_buffer("weighted_mask", weighted_mask)
        else:
            self.register_buffer("weighted_mask", None)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("You need to provide input_ids.")

        # Process input embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Forward pass
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(weight=self.weighted_mask)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LLMAgent(AbstractAgent):
    """Agent interface for LLMAgent baseline."""

    def __init__(
        self,
        config: LLMConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes LLM agent.
        :param config: global config of LLM agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._backbone = TransfuserBackbone(config)
        self._checkpoint_path = checkpoint_path
        self._model = SimplifiedLlamaForCausalLM.from_pretrained(BASE_MODEL)
        self._tokenizer = load_llama_tokenizer(BASE_MODEL)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

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
        
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        _, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        status_encoding = self._status_encoding(status_feature)
        sensor_token_str = " ".join(map(str, torch.flatten(bev_feature, 1).tolist()))
        status_token_str = " ".join(map(str, torch.flatten(status_encoding, 1).tolist()))

        prompt = f"""You are a certified professional driving instructor and please tell me step by step how to drive a car based on the input scenario.
        The sensor feature is the following: {sensor_token_str}
        The status feature is this: {status_token_str}"""

        input_ids = tokenize(self._tokenizer, prompt)["labels"]
        generated_ids = self._model.generate(input_ids=input_ids, max_length=100)
        decoded_token_ids = decode_generation_seqeunces(self._tokenizer, generated_ids)

        return {"pred_trajectory": torch.tensor([decoded_token_ids])}
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        target = targets["trajectory"]
        pred = predictions["pred_trajectory"]

        return F.l1_loss(pred, target)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
