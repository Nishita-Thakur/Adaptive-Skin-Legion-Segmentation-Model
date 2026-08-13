"""
bridge_diffusion.py — Brownian Bridge Diffusion Model (BDM).

The source image is RGB (3 channels), while the segmentation target is
a single-channel mask. The bridge itself is therefore represented in
single-channel space, while the U-Net can still receive a 3-channel
input as configured by the model YAML.

This keeps:
    RGB image -> DWT -> BDM -> 1-channel mask -> ECRF
"""

import torch
import torch.nn as nn

from .unet_backbone import UNetBackbone
from .boundary_head import BoundaryHead


class BrownianBridgeDiffusion(nn.Module):
    def __init__(
        self,
        unet_kwargs: dict,
        T_train: int = 1000,
        T_sample: int = 3,
        s_variance_scale: float = 1.0,
        boundary_head_enabled: bool = False,
        boundary_head_out_channels: int = 1,
    ):
        super().__init__()

        # The U-Net receives the configured RGB/DWT representation.
        # However, segmentation output is always one channel.
        unet_kwargs = dict(unet_kwargs)
        unet_kwargs["out_channels"] = 1

        self.unet = UNetBackbone(**unet_kwargs)

        self.T_train = T_train
        self.T_sample = T_sample
        self.s = s_variance_scale

        self.boundary_head_enabled = boundary_head_enabled

        if boundary_head_enabled:
            self.boundary_head = BoundaryHead(
                self.unet.final_feature_channels,
                out_channels=boundary_head_out_channels,
            )
        else:
            self.boundary_head = None

    # ---------------------------------------------------------
    # Convert source RGB image to the one-channel bridge space
    # ---------------------------------------------------------
    @staticmethod
    def image_to_bridge_space(x: torch.Tensor) -> torch.Tensor:
        """
        Convert an RGB source image to a single-channel image.

        If the input is already one-channel, it is returned unchanged.
        """
        if x.shape[1] == 1:
            return x

        if x.shape[1] == 3:
            # Standard luminance conversion.
            return (
                0.299 * x[:, 0:1]
                + 0.587 * x[:, 1:2]
                + 0.114 * x[:, 2:3]
            )

        # Generic fallback for unusual channel counts.
        return x.mean(dim=1, keepdim=True)

    # ---------------------------------------------------------
    # Forward-process quantities
    # ---------------------------------------------------------
    def m_t(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() / self.T_train

    def delta_t(self, t: torch.Tensor) -> torch.Tensor:
        m = self.m_t(t)
        return 2.0 * self.s * (m - m ** 2)

    def q_sample(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Sample x_t ~ q(x_t | x_0, x_T).

        Public API preserves the input channel count.
        Internally, the bridge uses a single-channel representation.
        """

        original_channels = x0.shape[1]

        # Convert both endpoints to the internal one-channel bridge space.
        x0_bridge = x0[:, :1] if original_channels > 1 else x0
        xT_bridge = self.image_to_bridge_space(xT)

        if noise is None:
            noise = torch.randn_like(x0)

        bridge_noise = noise[:, :1]

        m = self.m_t(t).view(-1, 1, 1, 1)
        delta = (
            self.delta_t(t)
            .clamp(min=1e-12)
            .view(-1, 1, 1, 1)
        )

        x_t_bridge = (
            x0_bridge
            + m * (xT_bridge - x0_bridge)
            + delta.sqrt() * bridge_noise
        )

        # Public API: return the same channel count as x0.
        if original_channels > 1:
            x_t = x_t_bridge.repeat(1, original_channels, 1, 1)
        else:
            x_t = x_t_bridge

        return x_t, noise

    def training_losses(
            self,
            x0_mask: torch.Tensor,
            xT_image: torch.Tensor,
            boundary_gt: torch.Tensor = None,
        ):
            B = x0_mask.shape[0]

            # Ensure segmentation target is one channel.
            if x0_mask.shape[1] != 1:
                x0_mask = x0_mask[:, :1]

            # Source image represented in bridge space.
            xT_bridge = self.image_to_bridge_space(xT_image)

            t = torch.randint(
                1,
                self.T_train + 1,
                (B,),
                device=x0_mask.device,
            )

            # One-channel Brownian bridge state.
            x_t, noise = self.q_sample(
                x0_mask,
                xT_bridge,
                t,
            )

            # The configured U-Net expects 3 channels.
            # Replicate the bridge state for the RGB-compatible input.
            unet_input_channels = self.unet.stem.in_channels

            if unet_input_channels == 1:
                model_input = x_t
            else:
                model_input = x_t.repeat(
                    1,
                    unet_input_channels,
                    1,
                    1,
                )

            # Predict a ONE-CHANNEL target.
            pred_mask_signal, decoder_features = self.unet(
                model_input,
                t,
            )

            m = self.m_t(t).view(-1, 1, 1, 1)
            delta = (
                self.delta_t(t)
                .clamp(min=1e-12)
                .view(-1, 1, 1, 1)
            )

            # q_sample returns noise in the public/input channel shape.
            # The bridge itself is single-channel, so use only the first
            # channel for the diffusion target.
            bridge_noise = noise[:, :1]

            target = (
                m * (xT_bridge - x0_mask)
                + delta.sqrt() * bridge_noise
            )

            diffusion_loss = torch.nn.functional.mse_loss(
                pred_mask_signal,
                target,
            )

            losses = {
                "diffusion_loss": diffusion_loss
            }

            if self.boundary_head_enabled and boundary_gt is not None:
                boundary_logits = self.boundary_head(
                    decoder_features
                )

                if boundary_logits.shape[-2:] != boundary_gt.shape[-2:]:
                    boundary_logits = torch.nn.functional.interpolate(
                        boundary_logits,
                        size=boundary_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                losses["boundary_logits"] = boundary_logits

            return losses, x_t

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        xT_image: torch.Tensor,
        num_steps: int = None,
        sigma: float = 0.0,
    ):
        """
        Generate a one-channel segmentation mask from the source image.
        """

        num_steps = num_steps or self.T_sample

        device = xT_image.device
        B = xT_image.shape[0]

        # Convert RGB source image to bridge space.
        xT_bridge = self.image_to_bridge_space(xT_image)

        # Timesteps {T, ..., 1}.
        seq = torch.linspace(
            self.T_train,
            1,
            num_steps,
            device=device,
        ).round().long()

        seq = torch.unique_consecutive(seq)

        # At t=T, bridge state corresponds to source image.
        x_t = xT_bridge.clone()

        boundary_logits = None
        decoder_features = None

        for i, t_val in enumerate(seq):

            t = torch.full(
                (B,),
                int(t_val.item()),
                device=device,
                dtype=torch.long,
            )

            # U-Net still receives the configured number of channels.
            unet_input_channels = self.unet.stem.in_channels

            if unet_input_channels == 1:
                model_input = x_t
            else:
                model_input = x_t.repeat(
                    1,
                    unet_input_channels,
                    1,
                    1,
                )

            pred_signal, decoder_features = self.unet(
                model_input,
                t,
            )

            m_t = self.m_t(t).view(
                -1, 1, 1, 1
            )

            delta_t = (
                self.delta_t(t)
                .clamp(min=1e-12)
                .view(-1, 1, 1, 1)
            )

            # Recover x0 estimate.
            x0_hat = (
                xT_bridge
                - pred_signal / m_t.clamp(min=1e-6)
            )

            x0_hat = x0_hat.clamp(-1.0, 1.0)

            if i == len(seq) - 1:
                x_t = x0_hat
                break

            t_prev = torch.full(
                (B,),
                int(seq[i + 1].item()),
                device=device,
                dtype=torch.long,
            )

            m_prev = self.m_t(t_prev).view(
                -1, 1, 1, 1
            )

            delta_prev = (
                self.delta_t(t_prev)
                .clamp(min=1e-12)
                .view(-1, 1, 1, 1)
            )

            if sigma > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            mean = (
                (1 - m_prev) * x0_hat
                + m_prev * xT_bridge
            )

            x_t = (
                mean
                + sigma * delta_prev.sqrt() * noise
            )

        if (
            self.boundary_head_enabled
            and decoder_features is not None
        ):
            boundary_logits = self.boundary_head(
                decoder_features
            )

        # Preserve the public API shape of sample().
        # The segmentation pipeline itself uses the first channel as the
        # one-channel mask, while direct BDM callers can still receive
        # the same number of channels as their input.
        if xT_image.shape[1] > 1:
            x_t = x_t.repeat(1, xT_image.shape[1], 1, 1)

        return x_t, boundary_logits