"""CPU check of the Transformers-backend tensor parallelism.

Each simulated rank (a thread) shards its copy of the same tiny random Llama
with ``_shard_for_tensor_parallel``, and must reproduce the logits of the
unsharded model. The all-reduce sums the ranks' tensors in process.

Run:  VLLM_NEURON_CPU_MODE=1 pytest -q test/unit/model/test_transformers_backend_tp.py
"""

import copy
import threading

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vllm_neuron.model.transformers.model import _shard_for_tensor_parallel


class _ThreadGroup:
    """The subset of vLLM's GroupCoordinator the sharding uses, over threads."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._slots: list[torch.Tensor | None] = [None] * world_size

    def rank_view(self, rank: int) -> "_RankGroup":
        return _RankGroup(self, rank)


class _RankGroup:
    def __init__(self, group: _ThreadGroup, rank: int):
        self._group = group
        self.rank_in_group = rank
        self.world_size = group.world_size

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        g = self._group
        g._slots[self.rank_in_group] = x
        g._barrier.wait()
        total = torch.stack(g._slots).sum(0)
        g._barrier.wait()
        return total


@pytest.mark.parametrize(
    "tp_size, num_kv_heads",
    [(2, 4), (4, 4), (4, 2), (8, 2)],  # the last two replicate KV heads
)
def test_sharded_logits_match(tp_size, num_kv_heads):
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=num_kv_heads,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=96,
        attention_bias=True,
        torch_dtype=torch.float32,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.arange(12).unsqueeze(0)
    with torch.no_grad():
        reference = model(input_ids).logits

    group = _ThreadGroup(tp_size)
    outputs, errors = [None] * tp_size, []

    def rank_main(rank: int) -> None:
        try:
            shard = copy.deepcopy(model)
            _shard_for_tensor_parallel(shard, group.rank_view(rank))
            with torch.no_grad():
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
        torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)
