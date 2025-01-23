from monai.networks.blocks import ResidualUnit
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.layers.convutils import same_padding
import numpy as np


class ResidualUnit_changed(ResidualUnit):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super(ResidualUnit_changed, self).__init__(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            subunits,
            adn_ordering,
            act,
            norm,
            dropout,
            dropout_dim,
            dilation,
            bias,
            last_conv_only,
            padding,
        )
        if not padding:
            padding = same_padding(kernel_size, dilation)
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if (
                np.prod(strides) == 1
            ):  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            # residual path is just avg pool
            self.residual = nn.AvgPool3d(
                kernel_size=rkernel_size,
                stride=strides,
                padding=rpadding,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations

        cx[:, : res.shape[1]] = (
            cx[:, : res.shape[1]] + res
        )  # add the residual to the output
        return cx
