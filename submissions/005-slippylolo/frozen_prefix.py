"""Batched frozen-prefix materialization for the pinned top-one Qwen tail."""

from __future__ import annotations

from dataclasses import dataclass

import torch


EXPECTED_DECODER_LAYERS = 28
EXPECTED_BOUNDARY_INDEX = 26
EXPECTED_HIDDEN_SIZE = 1_536
EXPECTED_SEQUENCE_LENGTH = 1_024


@dataclass(frozen=True)
class PrefixCacheMetrics:
    blocks: int
    bytes: int
    build_batch_blocks: int
    build_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocated_before_bytes: int
    reserved_before_bytes: int
    peak_allocated_increase_bytes: int
    peak_reserved_increase_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prefix_cache_blocks": self.blocks,
            "prefix_cache_bytes": self.bytes,
            "prefix_cache_build_batch_blocks": self.build_batch_blocks,
            "prefix_cache_build_seconds": self.build_seconds,
            "prefix_cache_peak_allocated_bytes": self.peak_allocated_bytes,
            "prefix_cache_peak_reserved_bytes": self.peak_reserved_bytes,
            "prefix_cache_allocated_before_bytes": self.allocated_before_bytes,
            "prefix_cache_reserved_before_bytes": self.reserved_before_bytes,
            "prefix_cache_peak_allocated_increase_bytes": (
                self.peak_allocated_increase_bytes
            ),
            "prefix_cache_peak_reserved_increase_bytes": (
                self.peak_reserved_increase_bytes
            ),
        }


class FrozenPrefixTailRunner:
    """Cache layer-26 outputs and execute layer 27 plus the final RMSNorm."""

    def __init__(
        self,
        *,
        transformer,
        boundary_index: int,
        sequence_length: int,
        materialization_batch_blocks: int | str,
    ) -> None:
        self.transformer = transformer
        self.layers = tuple(transformer.layers)
        self.boundary_index = boundary_index
        self.sequence_length = sequence_length
        self.materialization_batch_blocks = materialization_batch_blocks
        self.cache = None
        self.position_ids = None
        self.position_embeddings = None
        self.metrics = None
        self._validate_structure()

    def _validate_structure(self) -> None:
        config = self.transformer.config
        if len(self.layers) != EXPECTED_DECODER_LAYERS:
            raise RuntimeError("prefix cache requires exactly 28 decoder layers")
        if self.boundary_index != EXPECTED_BOUNDARY_INDEX:
            raise RuntimeError("prefix cache boundary must be decoder layer 26")
        if getattr(config, "hidden_size", None) != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError("prefix cache hidden width differs from pinned Qwen")
        if self.sequence_length != EXPECTED_SEQUENCE_LENGTH:
            raise RuntimeError("prefix cache sequence length must be exactly 1024")
        if getattr(config, "_attn_implementation", None) != "sdpa":
            raise RuntimeError("prefix cache requires the pinned SDPA implementation")
        layer_types = tuple(getattr(config, "layer_types", ()))
        if layer_types != ("full_attention",) * EXPECTED_DECODER_LAYERS:
            raise RuntimeError("prefix cache requires 28 full-attention decoder layers")
        if float(getattr(config, "attention_dropout", -1.0)) != 0.0:
            raise RuntimeError("prefix cache requires zero attention dropout")
        if not self.transformer.training:
            raise RuntimeError("prefix cache must preserve model train mode")
        stochastic_prefix = tuple(
            module
            for layer in self.layers[: self.boundary_index + 1]
            for module in layer.modules()
            if isinstance(module, torch.nn.Dropout)
            and module.training
            and float(module.p) != 0.0
        )
        if stochastic_prefix:
            raise RuntimeError("prefix cache cannot cross active stochastic dropout")
        if any(
            parameter.requires_grad
            for layer in self.layers[: self.boundary_index + 1]
            for parameter in layer.parameters()
        ):
            raise RuntimeError("prefix cache lower-layer parameters are not frozen")
        if not any(
            parameter.requires_grad
            for layer in self.layers[self.boundary_index + 1 :]
            for parameter in layer.parameters()
        ):
            raise RuntimeError("prefix cache top decoder layer is not trainable")
        if (
            self.materialization_batch_blocks != "all"
            and (
                isinstance(self.materialization_batch_blocks, bool)
                or not isinstance(self.materialization_batch_blocks, int)
                or self.materialization_batch_blocks < 1
            )
        ):
            raise RuntimeError("prefix cache build batch must be positive or 'all'")

    def _layer_forward(self, layer, hidden_states):
        return layer(
            hidden_states,
            attention_mask=None,
            position_ids=self.position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=self.position_embeddings,
        )

    def materialize(self, input_ids: torch.Tensor) -> PrefixCacheMetrics:
        if self.cache is not None:
            raise RuntimeError("prefix cache may be materialized only once")
        if (
            not torch.is_tensor(input_ids)
            or input_ids.dtype != torch.long
            or input_ids.device.type != "cuda"
            or input_ids.ndim != 2
            or input_ids.shape[0] < 1
            or input_ids.shape[1] != self.sequence_length
            or not input_ids.is_contiguous()
        ):
            raise RuntimeError("prefix cache input IDs violate the packed CUDA contract")

        blocks = input_ids.shape[0]
        batch_blocks = (
            blocks
            if self.materialization_batch_blocks == "all"
            else min(self.materialization_batch_blocks, blocks)
        )
        device = input_ids.device
        embed_tokens = self.transformer.embed_tokens
        if (
            embed_tokens.weight.device != device
            or embed_tokens.weight.dtype != torch.bfloat16
        ):
            raise RuntimeError("prefix cache embedding device/dtype differs")

        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()

        self.position_ids = torch.arange(
            self.sequence_length,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        first_stop = min(batch_blocks, blocks)
        with torch.no_grad():
            first_hidden = embed_tokens(input_ids[:first_stop])
            self.position_embeddings = self.transformer.rotary_emb(
                first_hidden,
                self.position_ids,
            )
        if (
            len(self.position_embeddings) != 2
            or any(
                value.device != device
                or value.dtype != torch.bfloat16
                or value.requires_grad
                for value in self.position_embeddings
            )
        ):
            raise RuntimeError("prefix cache rotary state violates the BF16 contract")

        # The stock Qwen forward creates no explicit tensor mask for this exact
        # no-padding, no-cache, full-attention SDPA case. Authenticate that before
        # bypassing the high-level transformer loop.
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=self.transformer.config,
            inputs_embeds=first_hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=self.position_ids,
        )
        if causal_mask is not None:
            raise RuntimeError("prefix cache expected SDPA to own the causal mask")

        self.cache = torch.empty(
            (
                blocks,
                self.sequence_length,
                EXPECTED_HIDDEN_SIZE,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        with torch.no_grad():
            for start in range(0, blocks, batch_blocks):
                stop = min(start + batch_blocks, blocks)
                hidden_states = (
                    first_hidden
                    if start == 0 and stop == first_stop
                    else embed_tokens(input_ids[start:stop])
                )
                for layer in self.layers[: self.boundary_index + 1]:
                    hidden_states = self._layer_forward(layer, hidden_states)
                if (
                    hidden_states.dtype != torch.bfloat16
                    or hidden_states.device != device
                    or hidden_states.shape
                    != (
                        stop - start,
                        self.sequence_length,
                        EXPECTED_HIDDEN_SIZE,
                    )
                ):
                    raise RuntimeError(
                        "prefix boundary output violates the BF16 CUDA shape contract"
                    )
                self.cache[start:stop].copy_(hidden_states)
        finished.record()
        finished.synchronize()
        del first_hidden

        if (
            self.cache.dtype != torch.bfloat16
            or self.cache.device != device
            or not self.cache.is_contiguous()
            or self.cache.requires_grad
            or self.cache.grad_fn is not None
        ):
            raise RuntimeError("materialized prefix cache violates its tensor contract")
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        metrics = PrefixCacheMetrics(
            blocks=blocks,
            bytes=self.cache.numel() * self.cache.element_size(),
            build_batch_blocks=batch_blocks,
            build_seconds=started.elapsed_time(finished) / 1000.0,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            allocated_before_bytes=allocated_before,
            reserved_before_bytes=reserved_before,
            peak_allocated_increase_bytes=max(
                0,
                peak_allocated - allocated_before,
            ),
            peak_reserved_increase_bytes=max(
                0,
                peak_reserved - reserved_before,
            ),
        )
        self.metrics = metrics
        return metrics

    def forward_suffix(self, start: int, stop: int) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("prefix cache has not been materialized")
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= self.cache.shape[0]
        ):
            raise RuntimeError("prefix cache slice is outside the tail horizon")
        hidden_states = self.cache[start:stop]
        hidden_states = self._layer_forward(
            self.layers[self.boundary_index + 1],
            hidden_states,
        )
        return self.transformer.norm(hidden_states)

    def close(self) -> None:
        self.cache = None
        self.position_ids = None
        self.position_embeddings = None
