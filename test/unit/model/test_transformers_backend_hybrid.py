"""CPU checks of the Transformers backend on a tiny LFM2-MoE.

Run:  VLLM_NEURON_CPU_MODE=1 pytest -q test/unit/model/test_transformers_backend_hybrid.py
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, Lfm2MoeConfig

from vllm_neuron.model.transformers.model import neuron_experts_forward


def tiny_config() -> Lfm2MoeConfig:
    return Lfm2MoeConfig(
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        num_dense_layers=1,
        num_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        conv_L_cache=3,
        dtype=torch.float32,
    )


def reference_model(config: Lfm2MoeConfig) -> torch.nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, experts_implementation="eager").eval()
    with torch.no_grad():  # the expert bias is zero-initialized: make it matter
        for module in model.modules():
            if hasattr(module, "expert_bias"):
                module.expert_bias.normal_()
    return model


@pytest.mark.parametrize("num_tokens", [2, 12])  # gathered (2*2*2 <= 8) and dense experts
@torch.no_grad()
def test_experts_paths_match_eager(num_tokens):
    reference = reference_model(tiny_config())
    experts = reference.model.layers[1].feed_forward.experts
    hidden = torch.randn(num_tokens, 64)
    top_k_index = torch.stack([torch.randperm(8)[:2] for _ in range(num_tokens)])
    top_k_weights = torch.rand(num_tokens, 2)
    expected = type(experts).forward.__wrapped__(experts, hidden, top_k_index, top_k_weights)
    out = neuron_experts_forward(experts, hidden, top_k_index, top_k_weights)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
