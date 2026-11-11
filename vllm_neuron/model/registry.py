# SPDX-License-Identifier: Apache-2.0
import logging
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM, PEagleLlamaForCausalLM
from .qwen3 import DFlashQwen3ForCausalLM, Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration
from .qwen3_asr import Qwen3ASRForConditionalGeneration
from .transformers import TransformersForCausalLM

logger = logging.getLogger(__name__)


def get_models() -> list[tuple[str, type]]:
    """Return a list of available model classes.

    Returns:
        list[tuple[str, type]]: A list of tuples containing model names and their corresponding classes.
            Each tuple contains (model_name, model_class) where:
            - model_name (str): The string identifier for the model, compatible with Hugging Face transformers architecture
            - model_class (type): The actual model class implementation
    """
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM),
        ("GptOssForCausalLM", GptOssForCausalLM),
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("PEagleLlamaForCausalLM", PEagleLlamaForCausalLM),
        ("DFlashDraftModel", DFlashQwen3ForCausalLM),
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
        ("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration),
        # Transformers modeling backend: what vLLM resolves to with
        # --model-impl transformers (and for architectures it has no
        # implementation of).
        ("TransformersForCausalLM", TransformersForCausalLM),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel
        from .synthetic.synthetic_stacked import SyntheticStackedNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))
        # SyntheticStackedNeuronModel — synthetic subclass with the GK stacked-KV
        # layout, for the non-GK-prefill -> GK-decode KV-cache adaptor DI test.
        models.append(("SyntheticStackedNeuronModel", SyntheticStackedNeuronModel))

    return models


def resolve_model_cls(model_config) -> type:
    """Resolve the class that runs ``model_config`` on Neuron.

    vLLM resolves an architecture that vLLM Neuron has no implementation of to
    vLLM's own (GPU) model class, which the Neuron runners cannot run. Serve
    those decoder models with the Transformers modeling backend instead, as
    vLLM does for the architectures it has no implementation of.
    """
    from vllm import ModelRegistry

    model_cls, arch = ModelRegistry.resolve_model_cls(
        model_config.architecture, model_config=model_config
    )
    if hasattr(model_cls, "from_configs") or model_config.runner_type != "generate":
        return model_cls
    import transformers

    hf_cls = getattr(transformers, arch, None)
    if hf_cls is not None and not hf_cls.is_backend_compatible():
        raise ValueError(
            f"vLLM Neuron has no implementation of {arch}, and its transformers "
            "implementation does not support custom attention functions, which the "
            "Transformers modeling backend needs."
        )
    logger.info(
        "vLLM Neuron has no implementation of %s: serving it with the Transformers "
        "modeling backend",
        arch,
    )
    return TransformersForCausalLM
