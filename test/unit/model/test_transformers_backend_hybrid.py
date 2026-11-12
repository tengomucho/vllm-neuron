"""CPU checks of the Transformers backend on an LFM2-MoE hybrid (short convolutions + attention).

The backend model runs prefill and batched decode steps over its paged caches
(attention K/V and short-convolution state) and must reproduce the logits of
the transformers model run on whole sequences.

Run:  VLLM_NEURON_CPU_MODE=1 pytest -q test/unit/model/test_transformers_backend_hybrid.py
"""

import copy
import threading

import pytest
import torch
from transformers import AutoModelForCausalLM, Lfm2MoeConfig

from vllm_neuron.model.transformers.model import (
    TransformersForCausalLM,
    _shard_for_tensor_parallel,
    _use_paged_short_convs,
    neuron_experts_forward,
)

from test_transformers_backend_tp import _ThreadGroup

BLOCK_SIZE, NUM_BLOCKS = 4, 16


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


def backend_model(reference: torch.nn.Module) -> TransformersForCausalLM:
    backend = TransformersForCausalLM(copy.deepcopy(reference.config), None)
    backend.model.float()  # the backend defaults to bf16
    backend.model.load_state_dict(reference.state_dict())
    _use_paged_short_convs(backend.model)
    caches = {}
    for layer in backend.get_kv_spec().layers:
        shape = (NUM_BLOCKS, layer.num_kv_heads, BLOCK_SIZE, layer.head_size)
        caches[layer.name] = [torch.zeros(shape) for _ in range(1 if layer.key_only else 2)]
    backend.bind_kv_cache(caches)
    return backend


def slots(blocks, start, n):
    pos = torch.arange(start, start + n)
    return torch.tensor(blocks)[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE


def step(backend, input_ids, positions, slot_mapping, block_table, max_query_len, sampling, kv_segment_size=0):
    meta = {
        "slot_mapping": slot_mapping,
        "block_table_tensor": block_table,
        "block_size": BLOCK_SIZE,
        "max_query_len": max_query_len,
        "decode_token_threshold": 1,
        "kv_segment_size": kv_segment_size,
    }
    attn_metadata = {layer.name: meta for layer in backend.get_kv_spec().layers}
    return backend(
        input_ids, positions, attn_metadata=attn_metadata, sampling_positions=torch.tensor(sampling)
    )


@torch.no_grad()
def test_prefill_then_batched_decode():
    reference = reference_model(tiny_config())
    backend = backend_model(reference)
    torch.manual_seed(1)
    seq_len, prompt_len, padded = 9, 6, 8
    seqs = {"a": [3, 7, 9], "b": [5, 11, 2]}
    tokens = {name: torch.randint(0, 96, (seq_len,)) for name in seqs}
    expected = {name: reference(ids[None]).logits[0] for name, ids in tokens.items()}

    # Prefill each sequence: real tokens first, then padding (slot -1).
    pad = padded - prompt_len
    for name, blocks in seqs.items():
        ids = torch.cat([tokens[name][:prompt_len], torch.zeros(pad, dtype=torch.long)])
        pos = torch.cat([torch.arange(prompt_len), torch.zeros(pad, dtype=torch.long)])
        sl = torch.cat([slots(blocks, 0, prompt_len), torch.full((pad,), -1)])
        logits = step(backend, ids, pos, sl, torch.tensor([blocks]), padded, list(range(prompt_len)))
        torch.testing.assert_close(logits, expected[name][:prompt_len], rtol=1e-4, atol=1e-4)

    # Decode both sequences, plus a padded batch row (block table -1, slot -1).
    block_table = torch.tensor([seqs["a"], seqs["b"], [-1, -1, -1]])
    for t in range(prompt_len, seq_len):
        ids = torch.stack([tokens["a"][t], tokens["b"][t], torch.tensor(0)])
        sl = torch.cat([slots(seqs["a"], t, 1), slots(seqs["b"], t, 1), torch.tensor([-1])])
        logits = step(backend, ids, torch.tensor([t, t, 0]), sl, block_table, 1, [0, 1, 2])
        torch.testing.assert_close(logits[0], expected["a"][t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(logits[1], expected["b"][t], rtol=1e-4, atol=1e-4)
        assert torch.isfinite(logits[2]).all()


@torch.no_grad()
def test_segmented_prefill():
    """A second prefill chunk reads the conv state and K/V of the first one."""
    reference = reference_model(tiny_config())
    backend = backend_model(reference)
    torch.manual_seed(2)
    blocks, ids = [4, 8, 1], torch.randint(0, 96, (10,))
    expected = reference(ids[None]).logits[0]
    step(backend, ids[:4], torch.arange(4), slots(blocks, 0, 4), torch.tensor([blocks]), 4, [3])
    logits = step(
        backend, ids[4:], torch.arange(4, 10), slots(blocks, 4, 6), torch.tensor([blocks]), 6,
        list(range(6)), kv_segment_size=8,
    )
    torch.testing.assert_close(logits, expected[4:], rtol=1e-4, atol=1e-4)


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


@pytest.mark.parametrize("tp_size", [2, 4])
@torch.no_grad()
def test_sharded_experts_logits_match(tp_size):
    """Experts sharded with the packed_colwise/rowwise/moe_tp_experts plan."""
    config = tiny_config()
    reference = reference_model(config)
    model = AutoModelForCausalLM.from_config(config, experts_implementation="neuron").eval()
    model.load_state_dict(reference.state_dict())
    input_ids = torch.arange(12).unsqueeze(0)
    expected = reference(input_ids).logits

    group = _ThreadGroup(tp_size)
    outputs, errors = [None] * tp_size, []

    def rank_main(rank: int) -> None:
        try:
            shard = copy.deepcopy(model)
            _shard_for_tensor_parallel(shard, group.rank_view(rank))
            outputs[rank] = shard(input_ids).logits
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            group._barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    for out in outputs:
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
