import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import torch
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class VLM(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(base_model_name)

    def forward(self, image_batch) -> list[str]:
        messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }]
            for img in image_batch
        ]
    
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda", dtype=torch.float32)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts

class LoRAWithMLP(nn.Module):
    def __init__(self, base_model_name: str, trajectory_sampling: TrajectorySampling, hidden_dim: int):
        super().__init__()

        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
        lora_config = LoraConfig(
            r=8,  # Rank of LoRA adaptation
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,  # Dropout for regularization
            target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
            bias="none")
        lora_model = get_peft_model(model, lora_config)
    
        self.base_model = lora_model
        self.trajectory_sampling = trajectory_sampling
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, self.trajectory_sampling.num_poses * 3),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Average pooling over sequence length (B, Seq, Hidden) → (B, Hidden)
        pooled_output = last_hidden_state.mean(dim=1)

        # Pass through extra MLP layers
        mlp_output = self.mlp(pooled_output)
        return mlp_output