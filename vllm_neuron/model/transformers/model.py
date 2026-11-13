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
  sinks. Segmented prefill (prompt chunks over the cache) runs vLLM Neuron's
  segmented attention, an NKI kernel. Single-step prefill and decode use
  PyTorch ops: the KV cache of a sequence is gathered from its block table on
  each decode step;
- LFM2-style hybrids, whose short-convolution layers keep their state in a
  paged cache too;
- mixture of experts, with the experts sharded over the ranks.
"""

import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface, AutoModelForCausalLM, PretrainedConfig
from transformers.integrations.moe import ExpertsInterface, batched_mm_experts_forward
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn import Sampler

ATTN_IMPLEMENTATION = "neuron_paged"
EXPERTS_IMPLEMENTATION = "neuron"


def _layer_name(layer_idx: int, kind: str = "self_attn") -> str:
    # The name the runner keys attn_metadata and the KV caches by.
    return f"layers.{layer_idx}.{kind}"


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
    num_seqs, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads, kv_len = key.shape[1], key.shape[2]
    # Each KV head attends for its group of query heads at once, rather than
    # K/V being repeated for every query head.
    grouped = query.reshape(num_seqs, num_kv_heads, -1, head_dim)  # [R, Hkv, G * Sq, D]
    scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * scaling
    scores = scores.view(num_seqs, num_q_heads, q_len, kv_len)
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
    if sinks is not None:
        # Per-head sink logit (gpt-oss): takes part in the softmax, adds no value.
        sink = sinks.float().view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    probs = probs.to(value.dtype).view(num_seqs, num_kv_heads, -1, kv_len)
    return torch.matmul(probs, value).view(num_seqs, num_q_heads, q_len, head_dim)


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
    - segmented prefill (one sequence, a prompt chunk after those in the
      cache): vLLM Neuron's segmented attention kernel, which reads the cache
      in segments of ``kv_segment_size`` tokens;
    - decode: each sequence attends over its cache, gathered through its block
      table, up to its own position.

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
    if not is_decode and meta.get("kv_segment_size") and softcap is None:
        out = NF.segmented_attention(
            query[0],
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=meta["block_table_tensor"][:1].to(torch.int32),  # one sequence
            prior_tokens=meta["cached_seq_len"],
            block_size=block_size,
            kv_segment_size=meta["kv_segment_size"],
            scale=scaling,
            sliding_window=sliding_window,
            sink=None if s_aux is None else s_aux.view(-1, 1),
            allow_vector_exp=False,
        )  # [Hq, T, D]
        return out.transpose(0, 1).unsqueeze(0), None

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


def _is_short_conv(module: nn.Module) -> bool:
    """An LFM2-style gated short convolution: ``in_proj`` -> B, C, x; ``C * conv(B * x)``."""
    return (
        getattr(module, "layer_idx", None) is not None
        and isinstance(getattr(module, "conv", None), nn.Conv1d)
        and isinstance(getattr(module, "in_proj", None), nn.Linear)
        and isinstance(getattr(module, "out_proj", None), nn.Module)
    )


def _conv_state_head_size(width: int, num_entries: int, block_size: int) -> int:
    """Row width of a key-only cache block that holds ``num_entries`` vectors of ``width``.

    A block has ``block_size`` rows. Each vector takes the largest number of
    rows that divides ``width`` and still fits ``num_entries`` vectors per block.
    """
    if block_size < num_entries:
        raise ValueError(
            f"A KV-cache block of {block_size} tokens cannot hold the "
            f"{num_entries} inputs of a short-convolution state."
        )
    rows = max(r for r in range(1, block_size // num_entries + 1) if width % r == 0)
    return width // rows


class _PagedShortConv(nn.Module):
    """An LFM2 short convolution whose state lives in the paged cache.

    The transformers module keeps a per-sequence conv state: the last
    ``kernel_size - 1`` inputs ``B * x`` of the convolution. Here the state goes
    to a key-only paged cache, which vLLM manages like a KV cache, through the
    sequence's block table. Each block keeps ``kernel_size - 1`` entries: the
    input at position ``p`` goes to entry ``p % (kernel_size - 1)`` of the block
    that holds ``p``. That is ``(kernel_size - 1) / block_size`` of the memory
    of a cache with one entry per token.

    A step writes the last ``kernel_size - 1`` tokens of each sequence. Each
    token takes its window (its predecessors and itself) from the step's own
    tokens, and reads from the cache only the positions before the step. Those
    are the last positions the sequence wrote, so no later position of their
    block has replaced them. Prefill, segmented prefill and batched decode all
    take this one path.
    """

    def __init__(self, short_conv: nn.Module):
        super().__init__()
        self.layer_idx = short_conv.layer_idx
        self.in_proj = short_conv.in_proj
        self.conv = short_conv.conv
        self.out_proj = short_conv.out_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        neuron_attn_metadata: dict | None = None,
        neuron_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        meta = neuron_attn_metadata[_layer_name(self.layer_idx, "conv")]
        b, c, x = self.in_proj(hidden_states[0]).chunk(3, dim=-1)  # [T, W] each
        bx = b * x
        num_tokens, width = bx.shape
        kernel_size = self.conv.kernel_size[0]
        num_entries = kernel_size - 1

        cache = self.conv_cache  # [num_blocks, 1, block_size, width // rows]
        num_blocks, _, block_size, row_width = cache.shape
        rows = width // row_width  # cache rows per entry
        if num_entries * rows > block_size:
            raise ValueError(
                f"The conv cache was laid out for another block size than {block_size}."
            )
        row_offsets = torch.arange(rows, device=bx.device)

        block_table = meta["block_table_tensor"]
        slot_mapping = meta["slot_mapping"]
        if meta["max_query_len"] <= meta["decode_token_threshold"]:
            # Decode: the tokens of each sequence are contiguous, one row each.
            seq_len = num_tokens // block_table.shape[0]
        else:
            # A prefill step schedules a single sequence.
            seq_len = num_tokens
        seq = torch.arange(num_tokens, device=bx.device) // seq_len
        positions = neuron_positions.view(-1)
        step_start = positions[seq * seq_len]  # first position of the sequence in the step

        def entry_rows(blocks: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            """The cache rows of the entry of ``pos`` in ``blocks``: [T, rows]."""
            first = blocks * block_size + (pos % num_entries) * rows
            return first.unsqueeze(-1) + row_offsets

        # Depthwise causal convolution; tap kernel_size - 1 is the token itself.
        weight = self.conv.weight[:, 0, :].float()  # [W, K]
        out = bx.float() * weight[:, -1]
        rows_view = cache.view(-1, row_width)
        for lag in range(1, kernel_size):
            pos = positions - lag
            in_step = torch.cat([bx.new_zeros(lag, width), bx])[:num_tokens]
            safe_pos = pos.clamp(min=0)
            blocks = block_table[seq, safe_pos // block_size].to(torch.long)
            # Unused block-table entries hold -1; their positions are not read.
            blocks = torch.where(blocks < 0, 0, blocks)
            cached = rows_view.index_select(0, entry_rows(blocks, safe_pos).reshape(-1))
            cached = cached.view(num_tokens, width) * (pos >= 0).unsqueeze(-1)
            value = torch.where((pos >= step_start).unsqueeze(-1), in_step, cached.to(bx.dtype))
            out = out + value.float() * weight[:, -1 - lag]
        if self.conv.bias is not None:
            out = out + self.conv.bias.float()

        # After the reads, keep the last num_entries tokens of each sequence:
        # the others go to the null block, so that no entry is written twice.
        # Padding writes to the null block 0 (or to slot -1).
        real = (slot_mapping >= block_size) & (slot_mapping < num_blocks * block_size)
        real_ahead = torch.cat([real[num_entries:], real.new_zeros(num_entries)])
        seq_ahead = torch.cat([seq[num_entries:], seq.new_full((num_entries,), -1)])
        last = real & ~(real_ahead[:num_tokens] & (seq_ahead[:num_tokens] == seq))
        blocks, _ = paged_slot_indices(
            torch.where(last, slot_mapping, -1), block_size, num_blocks
        )
        _write_kv(
            cache,
            bx.reshape(1, 1, num_tokens * rows, row_width),
            blocks.to(torch.long).repeat_interleave(rows),
            (entry_rows(torch.zeros_like(blocks), positions)).reshape(-1),
        )
        return self.out_proj(c * out.to(c.dtype)).unsqueeze(0)


def _use_paged_short_convs(model: nn.Module) -> None:
    """Replace the short convolutions of ``model`` with ``_PagedShortConv``."""
    for name, module in list(model.named_modules()):
        if _is_short_conv(module) and not isinstance(module, _PagedShortConv):
            parent_name, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent_name), attr, _PagedShortConv(module))


def neuron_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``ExpertsInterface`` function: gathered experts for a few tokens, all experts for many.

    transformers' ``batched_mm`` gathers the weights of each (token, expert)
    pair. That suits a decode step, but at prefill sizes the copies outgrow the
    experts themselves: a 4096-token prefill of LFM2-24B-A2B would gather ~77
    GB per layer at TP 4. Past half the expert count in pairs, every token goes
    through every expert instead, as two dense matmuls, the expert outputs
    scaled by the routing weights (zero for unrouted experts). Both paths have
    static shapes and the choice depends on the token count only, so each
    bucket compiles into a static graph.
    """
    num_tokens, top_k = top_k_index.shape
    num_experts = self.gate_up_proj.shape[0]
    if 2 * num_tokens * top_k <= num_experts:
        return batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    if not self.has_gate or self.has_bias or self.is_transposed or self.has_post_expert_norm:
        raise NotImplementedError(
            f"{type(self).__name__}: the dense experts path supports gated experts "
            "without bias, transposed weights or post-expert norm"
        )
    # [T, H] x [E * 2I, H] -> [T, E, 2I] -> gate -> [T, E, I]
    gate_up = F.linear(hidden_states, self.gate_up_proj.flatten(0, 1))
    act = self._apply_gate(gate_up.view(num_tokens, num_experts, -1))
    expert_ids = torch.arange(num_experts, device=hidden_states.device)
    routing = (
        (top_k_index.unsqueeze(-1) == expert_ids) * top_k_weights.unsqueeze(-1)
    ).sum(1)  # [T, E]
    act = act * routing.unsqueeze(-1).to(act.dtype)
    # [T, E * I] x [H, E * I] -> [T, H]: the sum over experts is in the matmul.
    down = self.down_proj.transpose(0, 1).flatten(1)
    return F.linear(act.flatten(1), down).to(hidden_states.dtype)


ExpertsInterface.register(EXPERTS_IMPLEMENTATION, neuron_experts_forward)


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


class _AllReduceOutput(nn.Module):
    """Sums the partial outputs of a module sharded over its input features.

    Wraps the experts of the ``moe_tp_experts`` style: each rank holds a slice
    of every expert's intermediate features, and the routing-weighted sum over
    experts is linear, so the partial sums only need adding once.
    """

    def __init__(self, module: nn.Module, tp_group):
        super().__init__()
        self.module = module
        self.tp_group = tp_group

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.tp_group.all_reduce(self.module(*args, **kwargs))


def _shard_expert_param(
    module: nn.Module, attr: str, style: str, rank: int, tp_size: int
) -> None:
    """Shard the 3D expert parameter ``module.<attr>``, [E, out, in] (or [E, in, out])."""
    param = getattr(module, attr)
    if param.dim() != 3:
        raise NotImplementedError(f"Tensor-parallel style {style!r} of {attr} {param.shape}")
    transposed = getattr(module, "is_transposed", False)
    out_dim, in_dim = (-1, -2) if transposed else (-2, -1)
    if style == "rowwise":
        shard = param.chunk(tp_size, dim=in_dim)[rank]
    elif style == "packed_colwise" and getattr(module, "is_concatenated", True):
        # [gate; up]: keep this rank's slice of each half.
        shard = torch.cat(
            [half.chunk(tp_size, dim=out_dim)[rank] for half in param.chunk(2, dim=out_dim)],
            dim=out_dim,
        )
    elif style in ("colwise", "packed_colwise"):
        # Interleaved [gate0, up0, gate1, up1, ...] slices like colwise.
        shard = param.chunk(tp_size, dim=out_dim)[rank]
    else:
        raise NotImplementedError(f"Tensor-parallel style {style!r} of {attr}")
    setattr(module, attr, nn.Parameter(shard.contiguous(), requires_grad=False))


def _shard_for_tensor_parallel(model: nn.Module, tp_group) -> None:
    """Shard ``model`` in place following its ``base_model_tp_plan``.

    The attention of transformers models reshapes its projections with
    ``view(..., -1, head_dim)``, so sharding the q/k/v projections by heads is
    enough for it to run on the local heads.
    """
    rank, tp_size = tp_group.rank_in_group, tp_group.world_size
    text_config = model.config.get_text_config()
    num_heads = text_config.num_attention_heads
    num_kv_heads = getattr(text_config, "num_key_value_heads", None) or num_heads
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // num_heads
    )
    if num_heads % tp_size:
        raise NotImplementedError(
            f"num_attention_heads={num_heads} is not divisible by "
            f"tensor_parallel_size={tp_size}"
        )
    # With fewer KV heads than ranks, replicate them, as vLLM does: each rank
    # keeps the KV head its query heads attend with.
    kv_replicas = 1
    if num_kv_heads % tp_size:
        if tp_size % num_kv_heads:
            raise NotImplementedError(
                f"num_key_value_heads={num_kv_heads} and "
                f"tensor_parallel_size={tp_size} must divide one another"
            )
        kv_replicas = tp_size // num_kv_heads
    # vLLM replaces some transformers config classes with its own (e.g.
    # lfm2_moe), which carry no plan: fall back to the model's config class.
    tp_plan = (
        getattr(text_config, "base_model_tp_plan", None)
        or getattr(model.base_model.config_class, "base_model_tp_plan", None)
        or {}
    )
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
    # Plan entries can name parameters: the 3D weights of fused experts.
    for name, _ in list(base_model.named_parameters()):
        style = next((s for p, s in patterns if p.match(name)), None)
        if style is not None:
            module_name, _, attr = name.rpartition(".")
            module = base_model.get_submodule(module_name)
            _shard_expert_param(module, attr, style, rank, tp_size)
    for name, module in list(base_model.named_modules()):
        style = next((s for p, s in patterns if p.match(name)), None)
        # Replicated modules (e.g. Qwen3's per-head q/k norms) need no change
        # for inference: the gradient all-reduce only matters in training.
        if style is None or style == "replicated_with_grad_allreduce":
            continue
        if style == "moe_tp_experts":
            if getattr(module, "has_bias", False):
                raise NotImplementedError(f"Tensor-parallel experts with bias ({name})")
            parent_name, _, attr = name.rpartition(".")
            parent = base_model.get_submodule(parent_name)
            setattr(parent, attr, _AllReduceOutput(module, tp_group))
            continue
        if not isinstance(module, nn.Linear) or style not in ("colwise", "rowwise"):
            raise NotImplementedError(
                f"Tensor-parallel style {style!r} of {name} ({type(module).__name__})"
            )
        parent_name, _, attr = name.rpartition(".")
        parent = base_model.get_submodule(parent_name)
        if style == "rowwise":
            setattr(parent, attr, _RowParallelLinear(module, rank, tp_size, tp_group))
        elif (
            kv_replicas > 1
            and type(parent).__name__.endswith("Attention")
            and module.out_features == num_kv_heads * head_dim
        ):
            # A k/v projection: keep the one KV head of this rank.
            _shard_colwise(module, rank // kv_replicas, num_kv_heads)
        else:
            _shard_colwise(module, rank, tp_size)


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


def _model_kwargs(hf_config: PretrainedConfig) -> dict:
    """The implementation kwargs of ``from_config`` and ``from_pretrained``."""
    kwargs = {"attn_implementation": ATTN_IMPLEMENTATION}
    text_config = hf_config.get_text_config()
    num_experts = next(
        (
            getattr(text_config, name)
            for name in ("num_experts", "num_local_experts", "n_routed_experts")
            if getattr(text_config, name, None)
        ),
        0,
    )
    if num_experts:
        # Mixture of experts: unlike the default grouped_mm (and its eager
        # fallback), neuron_experts_forward has shapes that do not depend on
        # the routing, so it compiles into a static graph.
        kwargs["experts_implementation"] = EXPERTS_IMPLEMENTATION
    return kwargs


class TransformersForCausalLM(nn.Module):
    """A transformers ``*ForCausalLM`` behind the vLLM Neuron runner interface."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        block_size: int = 32,
    ):
        super().__init__()
        self.config = hf_config
        self.neuron_config = neuron_config
        # The KV-cache block size, which lays out the short-convolution state.
        self.block_size = block_size
        # The dtype is read back from self.model rather than kept as an
        # attribute: Dynamo's precompile cache replaces, by object id, every
        # non-tensor module attribute it does not guard on, and a torch.dtype
        # singleton then corrupts the dtype of every tensor it serializes.
        self.model = AutoModelForCausalLM.from_config(
            hf_config,
            **_model_kwargs(hf_config),
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
        block_size = get_current_vllm_config().cache_config.block_size
        return cls(hf_config, neuron_config, block_size)

    def _cache_layers(self) -> dict[int, nn.Module]:
        """The modules that keep a paged cache, by layer index: one per decoder
        layer, either an attention or a short convolution."""
        layers = {}
        for module in self.model.modules():
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None and (
                type(module).__name__.endswith("Attention") or _is_short_conv(module)
            ):
                layers[layer_idx] = module
        num_layers = self.config.get_text_config().num_hidden_layers
        if sorted(layers) != list(range(num_layers)):
            raise ValueError(
                f"Found attention or short-conv layers {sorted(layers)} in "
                f"{type(self.model).__name__}, expected one per decoder layer "
                f"(0..{num_layers - 1})."
            )
        return layers

    def get_kv_spec(self) -> KVSpec:
        text_config = self.config.get_text_config()
        layers = []
        for layer_idx, module in sorted(self._cache_layers().items()):
            if _is_short_conv(module):
                # Key-only: the last kernel_size - 1 convolution inputs B * x
                # per block (see _PagedShortConv).
                head_size = _conv_state_head_size(
                    module.conv.in_channels, module.conv.kernel_size[0] - 1, self.block_size
                )
                layers.append(
                    LayerSpec(
                        name=_layer_name(layer_idx, "conv"),
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=self.model.dtype,
                        key_only=True,
                    )
                )
                continue
            head_dim = getattr(module, "head_dim", None) or (
                text_config.hidden_size // text_config.num_attention_heads
            )
            k_proj = getattr(module, "k_proj", None)
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
        for layer_idx, module in self._cache_layers().items():
            if _is_short_conv(module):
                (module.conv_cache,) = kv_caches[_layer_name(layer_idx, "conv")]
            else:
                module.k_cache, module.v_cache = kv_caches[_layer_name(layer_idx)]

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        # transformers loads the checkpoint, with its own key renames and
        # weight conversions; the runner moves the model to the device after.
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=self.config,
            **_model_kwargs(self.config),
            dtype=self.model.dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            _shard_for_tensor_parallel(self.model, tp_group)
        _use_paged_short_convs(self.model)

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
