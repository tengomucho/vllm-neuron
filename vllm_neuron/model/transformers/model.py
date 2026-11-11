# SPDX-License-Identifier: Apache-2.0
"""Transformers modeling backend for vLLM Neuron.

Serves a Hugging Face ``*ForCausalLM`` with the modeling code of transformers,
as vLLM's Transformers modeling backend does on GPU (``--model-impl
transformers``). transformers builds the model and loads its checkpoint; vLLM
Neuron provides the paged KV cache and the attention over it, through a
function registered in the transformers ``AttentionInterface``. The runner
compiles the whole forward with ``torch.compile``, as for the native models.

Prototype scope:

- decoder-only text models whose attention goes through the
  ``AttentionInterface`` (``model_class.is_backend_compatible()``);
- tensor parallelism from the model's ``base_model_tp_plan`` (``colwise`` and
  ``rowwise`` linear layers; embeddings and LM head stay replicated);
- causal attention, optionally sliding-window, soft-capped, or with attention
  sinks, computed with PyTorch ops (no NKI kernels): the KV cache of a
  sequence is gathered from its block table on each decode step.
"""

import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface, AutoModelForCausalLM, PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn import Sampler

ATTN_IMPLEMENTATION = "neuron_paged"


def _layer_name(layer_idx: int) -> str:
    # The name the runner keys attn_metadata and the KV caches by.
    return f"layers.{layer_idx}.self_attn"


def _write_kv(
    cache: torch.Tensor,
    states: torch.Tensor,
    block_indices: torch.Tensor,
    position_indices: torch.Tensor,
) -> None:
    """Scatter ``states`` [1, H, T, D] into ``cache`` [num_blocks, H, block_size, D]."""
    _, num_heads, num_tokens, head_dim = states.shape
    head_indices = torch.arange(
        num_heads, dtype=torch.long, device=states.device
    ).repeat_interleave(num_tokens)
    cache.index_put_(
        (
            block_indices.repeat(num_heads),
            head_indices,
            position_indices.repeat(num_heads),
        ),
        states[0].reshape(num_heads * num_tokens, head_dim).to(cache.dtype),
    )


def _gather_kv(cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Gather each sequence's cache: [num_blocks, H, BS, D] -> [R, H, MB * BS, D]."""
    num_seqs, max_blocks = block_table.shape
    _, num_heads, block_size, head_dim = cache.shape
    # Unused entries hold -1 (read-side sentinel): point them at block 0. The
    # attention mask drops their positions.
    block_ids = torch.where(block_table < 0, 0, block_table).reshape(-1).to(torch.long)
    gathered = cache.index_select(0, block_ids)
    return (
        gathered.view(num_seqs, max_blocks, num_heads, block_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_seqs, num_heads, max_blocks * block_size, head_dim)
    )


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    scaling: float,
    softcap: float | None,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Masked attention. query [R, Hq, Sq, D], key/value [R, Hkv, Sk, D], allowed [R, Sq, Sk]."""
    num_groups = query.shape[1] // key.shape[1]
    if num_groups > 1:
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)).float() * scaling
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
    if sinks is not None:
        # Per-head sink logit (gpt-oss): takes part in the softmax, adds no value.
        sink = sinks.float().view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(value.dtype), value)


def neuron_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    softcap: float | None = None,
    s_aux: torch.Tensor | None = None,
    neuron_attn_metadata: dict | None = None,
    neuron_positions: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """``AttentionInterface`` function over the vLLM Neuron paged KV cache.

    transformers calls it with the projected, rotated states of all the tokens
    of the step flattened into a batch of one: query [1, Hq, T, D], key and
    value [1, Hkv, T, D]. It writes key and value to the paged cache at the
    step's slots, then attends:

    - prefill (one sequence, no prior cache): causal attention over the step's
      own tokens; the padding the runner appends comes last, so no real token
      sees it;
    - decode, or segmented prefill: each sequence attends over its cache,
      gathered through its block table, up to its own position.

    Returns the attention output [1, T, Hq, D].
    """
    meta = neuron_attn_metadata[_layer_name(module.layer_idx)]
    _, num_q_heads, num_tokens, head_dim = query.shape
    if scaling is None:
        scaling = head_dim**-0.5

    k_cache, v_cache = module.k_cache, module.v_cache
    block_size = meta["block_size"]
    block_indices, position_indices = paged_slot_indices(
        meta["slot_mapping"], block_size, k_cache.shape[0]
    )
    _write_kv(k_cache, key, block_indices, position_indices)
    _write_kv(v_cache, value, block_indices, position_indices)

    is_decode = meta["max_query_len"] <= meta["decode_token_threshold"]
    if is_decode or meta.get("kv_segment_size"):
        block_table = meta["block_table_tensor"]
        if not is_decode:
            # A prefill step schedules a single sequence.
            block_table = block_table[:1]
        num_seqs = block_table.shape[0]
        seq_len = num_tokens // num_seqs
        q = query[0].view(num_q_heads, num_seqs, seq_len, head_dim).transpose(0, 1)
        k = _gather_kv(k_cache, block_table).to(query.dtype)
        v = _gather_kv(v_cache, block_table).to(query.dtype)
        q_pos = neuron_positions.view(num_seqs, seq_len, 1)
        k_pos = torch.arange(k.shape[2], device=query.device).view(1, 1, -1)
    else:
        num_seqs, seq_len = 1, num_tokens
        q, k, v = query, key, value
        # Causal by token index: the runner pads after the real tokens, and
        # with no prior cache index and position agree on the real tokens.
        q_pos = torch.arange(num_tokens, device=query.device).view(1, -1, 1)
        k_pos = q_pos.view(1, 1, -1)

    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed = allowed & (k_pos > q_pos - sliding_window)

    out = _attend(q, k, v, allowed, scaling, softcap, s_aux)  # [R, Hq, S, D]
    out = out.transpose(0, 1).reshape(num_q_heads, num_tokens, head_dim)
    return out.transpose(0, 1).unsqueeze(0), None


AttentionInterface.register(ATTN_IMPLEMENTATION, neuron_paged_attention_forward)


class _RowParallelLinear(nn.Module):
    """An ``nn.Linear`` sharded over its input features: all-reduces its output."""

    def __init__(self, linear: nn.Linear, rank: int, tp_size: int, tp_group):
        super().__init__()
        shard = linear.in_features // tp_size
        self.weight = nn.Parameter(
            linear.weight[:, rank * shard : (rank + 1) * shard].contiguous(),
            requires_grad=False,
        )
        # Added once, after the reduction.
        self.bias = linear.bias
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tp_group.all_reduce(F.linear(x, self.weight))
        return out if self.bias is None else out + self.bias


def _shard_colwise(linear: nn.Linear, rank: int, tp_size: int) -> None:
    """Keep this rank's slice of the output features of ``linear``."""
    shard = linear.out_features // tp_size
    rows = slice(rank * shard, (rank + 1) * shard)
    linear.weight = nn.Parameter(linear.weight[rows].contiguous(), requires_grad=False)
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[rows].contiguous(), requires_grad=False)
    linear.out_features = shard


def _shard_for_tensor_parallel(model: nn.Module, tp_group) -> None:
    """Shard ``model`` in place following its ``base_model_tp_plan``.

    The attention of transformers models reshapes its projections with
    ``view(..., -1, head_dim)``, so sharding the q/k/v projections by heads is
    enough for it to run on the local heads.
    """
    rank, tp_size = tp_group.rank_in_group, tp_group.world_size
    text_config = model.config.get_text_config()
    for name in ("num_attention_heads", "num_key_value_heads"):
        value = getattr(text_config, name, None)
        if value is not None and value % tp_size:
            raise NotImplementedError(
                f"{name}={value} is not divisible by tensor_parallel_size={tp_size}"
            )
    tp_plan = getattr(text_config, "base_model_tp_plan", None) or {}
    if not tp_plan:
        raise NotImplementedError(
            f"{type(model).__name__} has no base_model_tp_plan: tensor parallelism "
            "is not supported for it"
        )
    patterns = [
        (re.compile("^" + re.escape(key).replace(r"\*", r"\d+") + "$"), style)
        for key, style in tp_plan.items()
    ]
    base_model = model.base_model
    for name, module in list(base_model.named_modules()):
        style = next((s for p, s in patterns if p.match(name)), None)
        # Replicated modules (e.g. Qwen3's per-head q/k norms) need no change
        # for inference: the gradient all-reduce only matters in training.
        if style is None or style == "replicated_with_grad_allreduce":
            continue
        if not isinstance(module, nn.Linear) or style not in ("colwise", "rowwise"):
            raise NotImplementedError(
                f"Tensor-parallel style {style!r} of {name} ({type(module).__name__})"
            )
        if style == "colwise":
            _shard_colwise(module, rank, tp_size)
        else:
            parent_name, _, attr = name.rpartition(".")
            parent = base_model.get_submodule(parent_name)
            setattr(parent, attr, _RowParallelLinear(module, rank, tp_size, tp_group))


def _model_dtype(hf_config: PretrainedConfig) -> torch.dtype:
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None and isinstance(
        vllm_config.model_config.dtype, torch.dtype
    ):
        return vllm_config.model_config.dtype
    dtype = getattr(hf_config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16


class TransformersForCausalLM(nn.Module):
    """A transformers ``*ForCausalLM`` behind the vLLM Neuron runner interface."""

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None):
        super().__init__()
        self.config = hf_config
        self.neuron_config = neuron_config
        # The dtype is read back from self.model rather than kept as an
        # attribute: Dynamo's precompile cache replaces, by object id, every
        # non-tensor module attribute it does not guard on, and a torch.dtype
        # singleton then corrupts the dtype of every tensor it serializes.
        self.model = AutoModelForCausalLM.from_config(
            hf_config,
            attn_implementation=ATTN_IMPLEMENTATION,
            dtype=_model_dtype(hf_config),
        )
        self.model.eval()

        ods_config = neuron_config.on_device_sampling_config if neuron_config else None
        self.sampler = Sampler(ods_config) if ods_config is not None else None
        self._gather_logits = neuron_config is not None and (
            neuron_config.max_logprobs != 0
            or neuron_config.debug_logits_dir is not None
        )

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ):
        return cls(hf_config, neuron_config)

    def _attention_layers(self) -> dict[int, nn.Module]:
        """The modules that call the attention function, by layer index."""
        layers = {}
        for module in self.model.modules():
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and type(module).__name__.endswith("Attention"):
                layers[layer_idx] = module
        num_layers = self.config.get_text_config().num_hidden_layers
        if sorted(layers) != list(range(num_layers)):
            raise ValueError(
                f"Found attention layers {sorted(layers)} in {type(self.model).__name__}, "
                f"expected one per decoder layer (0..{num_layers - 1})."
            )
        return layers

    def get_kv_spec(self) -> KVSpec:
        text_config = self.config.get_text_config()
        layers = []
        for layer_idx, attn in sorted(self._attention_layers().items()):
            head_dim = getattr(attn, "head_dim", None) or (
                text_config.hidden_size // text_config.num_attention_heads
            )
            k_proj = getattr(attn, "k_proj", None)
            if isinstance(k_proj, nn.Linear):
                num_kv_heads = k_proj.out_features // head_dim
            else:
                num_kv_heads = getattr(text_config, "num_key_value_heads", None) or (
                    text_config.num_attention_heads
                )
            # Sliding-window layers get a full-length cache: the attention
            # applies the window itself.
            layers.append(
                LayerSpec(
                    name=_layer_name(layer_idx),
                    num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                    dtype=self.model.dtype,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer_idx, attn in self._attention_layers().items():
            attn.k_cache, attn.v_cache = kv_caches[_layer_name(layer_idx)]

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        # transformers loads the checkpoint, with its own key renames and
        # weight conversions; the runner moves the model to the device after.
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=self.config,
            attn_implementation=ATTN_IMPLEMENTATION,
            dtype=self.model.dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            _shard_for_tensor_parallel(self.model, tp_group)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ):
        if spec_decode_metadata is not None:
            raise NotImplementedError("Speculative decoding is not supported yet.")
        positions = positions.to(torch.long)
        output = self.model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            logits_to_keep=sampling_positions,
            neuron_attn_metadata=attn_metadata,
            neuron_positions=positions,
        )
        logits = output.logits[0]  # [num_reqs, vocab_size]
        if self.sampler is None:
            return logits
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        return sampled_tokens, logits if self._gather_logits else None
