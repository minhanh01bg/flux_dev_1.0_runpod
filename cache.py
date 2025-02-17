
import torch
from diffusers import FluxPipeline

from huggingface_hub import login
login(token="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")

pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
