"""Load the complete checkpoint strictly, including its frozen base weights."""
import torch
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


class Organism(nn.Module):
    def __init__(self, config_dict, tokenizer):
        super().__init__()
        config = Qwen3Config.from_dict(config_dict)
        config.vocab_size = len(tokenizer)
        config._attn_implementation = 'sdpa'
        # The checkpoint includes every base, adapter and projection tensor.
        # Meta construction avoids downloading or allocating a second base model.
        with torch.device('meta'):
            base = Qwen3ForCausalLM(config)
            self.codi = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=128, lora_alpha=32, lora_dropout=.1, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']))
            d = config.hidden_size
            self.prj = nn.Sequential(nn.Dropout(0.), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.prj.add_module('ln', nn.LayerNorm(d))
        # RoPE frequencies are nonpersistent buffers, absent from state_dict.
        self.codi.get_base_model().model.rotary_emb = Qwen3RotaryEmbedding(config, device='cpu')
        self.tokenizer = tokenizer
        self.use_prj = True


def load(checkpoint, config_dict):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = Organism(config_dict, tokenizer)
    state = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device='cuda', dtype=torch.bfloat16).eval()
