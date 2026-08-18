"""Cross-sensor segmentation baselines compatible with ``train_pure_seg.py``.

All models consume the mixed-loader interface used by :class:`MixedS2L8Dataset`::

    forward(x, sensor_types) -> {"msi_seg": [B, 1, H, W]}

``x`` always has 11 channels. Landsat-8 samples occupy the first seven channels
and are zero padded by the data loader.  The baselines deliberately keep the
decoder, segmentation head, loss and training schedule compatible with the
existing routed model so that the comparison primarily tests sensor handling.

The ``spectral_set`` variant is a local, from-scratch SEnSeI-style control.  It
uses a shared wavelength-conditioned set encoder, but it is not a drop-in copy
of the authors' pretrained SEnSeI implementation and should be reported as a
reimplementation/inspired baseline.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from running.routed_unet import RoutedEncoder, SegmentationHead


# Physical channel order used by the current datasets.
# S2: B1--B8, B8A, B11, B12 (11 channels)
# L8: SR_B1--SR_B7 (7 channels)
S2_COMMON_INDICES = (0, 1, 2, 3, 7, 9, 10)
L8_COMMON_INDICES = (0, 1, 2, 3, 4, 5, 6)


def _sensor_indices(sensor_types: Sequence[str], sensor: str) -> List[int]:
    """Return sample indices for one sensor with tolerant input handling."""
    if torch.is_tensor(sensor_types):
        # Optional future numeric convention: 0=S2, 1=L8.
        target = 0 if sensor == "s2" else 1
        return (sensor_types.detach().cpu() == target).nonzero(as_tuple=False).flatten().tolist()
    return [i for i, value in enumerate(sensor_types) if str(value).lower() == sensor]


def _unique_parameters(modules: Iterable[nn.Module]):
    """Yield parameters once even when modules share weights."""
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            key = id(parameter)
            if key not in seen:
                seen.add(key)
                yield parameter


class SharedPixelShuffleDecoder(nn.Module):
    """The existing four-stage shared PixelShuffle decoder and segmentation head."""

    def __init__(self, hidden_dim: int, mamba_dim: int = 64):
        super().__init__()
        encoder_dims = [128, 256, 512, hidden_dim]
        decoder_dims = encoder_dims[::-1]
        skip_channels = [mamba_dim] + encoder_dims[:-1]

        self.skip_align = nn.ModuleList(
            nn.Conv2d(skip_channels[-(i + 1)], decoder_dims[i], 1)
            for i in range(len(decoder_dims) - 1)
        )
        self.decoder_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(decoder_dims[i] * 2, decoder_dims[i + 1], 3, padding=1),
                nn.GroupNorm(8, decoder_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for i in range(len(decoder_dims) - 1)
        )
        self.pixel_shuffle_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GroupNorm(8, dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for dim in decoder_dims
        )
        self.seg_head = SegmentationHead(decoder_dims[-1])

    def forward(self, feat: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        x = feat
        for i, upsample in enumerate(self.pixel_shuffle_layers):
            x = upsample(x)
            if i < len(self.decoder_convs):
                skip = skips[-(i + 1)]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
                skip = self.skip_align[i](skip)
                x = self.decoder_convs[i](torch.cat([x, skip], dim=1))
        return self.seg_head(x)


class SharedBackboneBaseline(nn.Module):
    """Base class for grouped sensor routing through one shared encoder/decoder."""

    def __init__(
        self,
        shared_in_channels: int,
        hidden_dim: int = 512,
        num_mamba_layers: int = 2,
        num_topk_layers: int = 2,
        mamba_dim: int = 64,
        topk_rate: float = 0.4,
        use_pixel_unshuffle: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mamba_dim = mamba_dim
        self.shared_encoder = RoutedEncoder(
            shared_in_channels,
            [128, 256, 512, hidden_dim],
            num_mamba=num_mamba_layers,
            num_topk=num_topk_layers,
            mamba_dim=mamba_dim,
            topk_rate=topk_rate,
            use_pixel_unshuffle=use_pixel_unshuffle,
        )
        self.shared_decoder = SharedPixelShuffleDecoder(hidden_dim, mamba_dim=mamba_dim)

    def _prepare_sensor_input(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        raise NotImplementedError

    def _input_modules(self) -> Sequence[nn.Module]:
        return ()

    def encoder_parameters(self):
        """Parameter iterator used by the existing two-group AdamW setup."""
        return _unique_parameters([*self._input_modules(), self.shared_encoder])

    def set_epoch(self, epoch: int, total_epochs: int):
        # These controls do not use annealed ECA; retain the common trainer API.
        del epoch, total_epochs

    def _forward_sensor(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        prepared = self._prepare_sensor_input(x, sensor)
        feat, skips = self.shared_encoder(prepared)
        return self.shared_decoder(feat, skips)

    def forward(self, x: torch.Tensor, sensor_types: Sequence[str]):
        if len(sensor_types) != x.size(0):
            raise ValueError(
                f"sensor_types length ({len(sensor_types)}) must match batch size ({x.size(0)})"
            )

        output = x.new_zeros((x.size(0), 1, x.size(2), x.size(3)))
        for sensor in ("s2", "l8"):
            indices = _sensor_indices(sensor_types, sensor)
            if not indices:
                continue
            index = torch.as_tensor(indices, device=x.device, dtype=torch.long)
            prediction = self._forward_sensor(x.index_select(0, index), sensor)
            output = output.index_copy(0, index, prediction)
        return {"msi_seg": output}


class CommonBandSharedUNet(SharedBackboneBaseline):
    """Seven physically corresponding bands followed by one fully shared network."""

    def __init__(self, **kwargs):
        super().__init__(shared_in_channels=7, **kwargs)

    def _prepare_sensor_input(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        indices = S2_COMMON_INDICES if sensor == "s2" else L8_COMMON_INDICES
        index = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        return x.index_select(1, index)


class SensorAdapterSharedUNet(SharedBackboneBaseline):
    """Sensor-private input projections followed by a completely shared backbone.

    This is the clean architectural control for the adapter/projector pattern used
    by models such as MultiMAE and AnySat, without importing pretrained weights.
    """

    def __init__(self, adapter_dim: int = 64, **kwargs):
        if adapter_dim % 8 != 0:
            raise ValueError("adapter_dim must be divisible by 8 for GroupNorm")
        super().__init__(shared_in_channels=adapter_dim, **kwargs)
        self.s2_adapter = nn.Sequential(
            nn.Conv2d(11, adapter_dim, 3, padding=1),
            nn.GroupNorm(8, adapter_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l8_adapter = nn.Sequential(
            nn.Conv2d(7, adapter_dim, 3, padding=1),
            nn.GroupNorm(8, adapter_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _input_modules(self) -> Sequence[nn.Module]:
        return (self.s2_adapter, self.l8_adapter)

    def _prepare_sensor_input(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        if sensor == "s2":
            return self.s2_adapter(x[:, :11])
        return self.l8_adapter(x[:, :7])


class SpectralSetEncoder(nn.Module):
    """Permutation-invariant wavelength-conditioned multispectral set encoder.

    Nominal centre wavelengths and bandwidths are expressed in nanometres.  A
    single descriptor MLP is shared by both sensors, and reflectance-weighted band
    embeddings are averaged into a fixed-width spatial feature map.
    """

    DEFAULT_METADATA: Mapping[str, Sequence[Sequence[float]]] = {
        "s2": (
            (443, 21), (490, 66), (560, 36), (665, 31), (705, 15), (740, 15),
            (783, 20), (842, 106), (865, 21), (1610, 91), (2190, 175),
        ),
        "l8": (
            (443, 20), (482, 60), (562, 60), (655, 30), (865, 30),
            (1609, 80), (2201, 180),
        ),
    }

    def __init__(self, out_channels: int = 64, metadata=None):
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError("out_channels must be divisible by 8 for GroupNorm")
        metadata = metadata or self.DEFAULT_METADATA
        for sensor in ("s2", "l8"):
            descriptor = torch.tensor(metadata[sensor], dtype=torch.float32)
            descriptor[:, 0] = (descriptor[:, 0] - 400.0) / (2500.0 - 400.0)
            descriptor[:, 1] = descriptor[:, 1] / 200.0
            self.register_buffer(f"{sensor}_descriptor", descriptor, persistent=True)

        self.band_mlp = nn.Sequential(
            nn.Linear(2, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        descriptor = getattr(self, f"{sensor}_descriptor")[: x.size(1)]
        band_embedding = self.band_mlp(descriptor)
        global_context = band_embedding.mean(dim=0, keepdim=True).expand_as(band_embedding)
        band_embedding = self.context_mlp(torch.cat([band_embedding, global_context], dim=-1))
        # The +0.5 offset preserves a descriptor signal for low-reflectance pixels,
        # matching the band-multiplication motivation of SEnSeI-style encoders.
        feature = torch.einsum("bchw,cd->bdhw", x + 0.5, band_embedding)
        feature = feature / max(1, x.size(1))
        return self.output(feature)


class SpectralSetSharedUNet(SharedBackboneBaseline):
    """SEnSeI-style spectral set encoder followed by one shared spatial network."""

    def __init__(self, adapter_dim: int = 64, spectral_metadata=None, **kwargs):
        super().__init__(shared_in_channels=adapter_dim, **kwargs)
        self.spectral_encoder = SpectralSetEncoder(adapter_dim, metadata=spectral_metadata)

    def _input_modules(self) -> Sequence[nn.Module]:
        return (self.spectral_encoder,)

    def _prepare_sensor_input(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        channels = 11 if sensor == "s2" else 7
        return self.spectral_encoder(x[:, :channels], sensor)


def build_multisensor_model(
    model_name: str,
    *,
    hidden_dim: int = 512,
    num_mamba_layers: int = 2,
    num_topk_layers: int = 2,
    mamba_dim: int = 64,
    topk_rate: float = 0.4,
    adapter_dim: int = 64,
    shared_pixel_unshuffle: bool = False,
) -> nn.Module:
    """Build a local from-scratch multi-sensor comparison model."""
    common_kwargs = dict(
        hidden_dim=hidden_dim,
        num_mamba_layers=num_mamba_layers,
        num_topk_layers=num_topk_layers,
        mamba_dim=mamba_dim,
        topk_rate=topk_rate,
        use_pixel_unshuffle=shared_pixel_unshuffle,
    )
    if model_name == "common_shared":
        return CommonBandSharedUNet(**common_kwargs)
    if model_name == "sensor_adapter":
        return SensorAdapterSharedUNet(adapter_dim=adapter_dim, **common_kwargs)
    if model_name == "spectral_set":
        return SpectralSetSharedUNet(adapter_dim=adapter_dim, **common_kwargs)
    raise ValueError(
        f"Unknown multi-sensor baseline: {model_name}. "
        "Expected common_shared, sensor_adapter, or spectral_set."
    )


__all__ = [
    "CommonBandSharedUNet",
    "SensorAdapterSharedUNet",
    "SpectralSetSharedUNet",
    "SpectralSetEncoder",
    "build_multisensor_model",
]
