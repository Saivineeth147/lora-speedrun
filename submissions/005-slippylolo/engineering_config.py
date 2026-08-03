"""Immutable combined implementation for the selected Track 1 candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ENGINEERING_VARIANT = "combined"


@dataclass(frozen=True)
class EngineeringFeatures:
    true_prefix_freeze: bool
    direct_static_lora: bool
    parent_layer_dispatch: bool

    @property
    def direct_adapter_writer(self) -> bool:
        return self.direct_static_lora

    @property
    def direct_packed_parent_runtime(self) -> bool:
        return self.direct_static_lora and self.parent_layer_dispatch

    def to_dict(self) -> dict[str, bool]:
        return {
            **asdict(self),
            "direct_adapter_writer": self.direct_adapter_writer,
            "direct_packed_parent_runtime": self.direct_packed_parent_runtime,
        }


ENGINEERING_FEATURES = EngineeringFeatures(
    true_prefix_freeze=True,
    direct_static_lora=True,
    parent_layer_dispatch=True,
)
