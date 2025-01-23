import torch
import torch.nn as nn
from nets.residual_block import ResidualUnit_changed as ResidualUnit
from monai.networks.blocks import Convolution
from collections.abc import Sequence
from monai.networks.layers.convutils import same_padding
from torchinfo import summary


class MOEConvolution(Convolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context):
        x = self.conv(x)
        if hasattr(self, "adn"):
            x = self.adn(x)
        return x


class MOESequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context):
        for module in self:
            x = module(x, context)
        return x


class MOEUpsample(nn.Upsample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context):
        return super().forward(x)


class MoEBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        num_experts,
        gate_input_dim,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate_input_dim = gate_input_dim
        self.experts = nn.ModuleList(
            [
                Convolution(
                    spatial_dims,
                    in_channels,
                    out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    adn_ordering=adn_ordering,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    dropout_dim=dropout_dim,
                    dilation=dilation,
                    bias=bias,
                    conv_only=conv_only,
                    padding=padding,
                )
                for _ in range(num_experts)
            ]
        )
        self.gating_network = nn.Linear(gate_input_dim, num_experts)

    def forward(self, x, context):
        B = context.shape[0]
        context = context.view(B, -1)
        assert (
            context.shape[1] == self.gate_input_dim
        )  # Check if context has the right shape
        gating_weights = self.gating_network(context)
        gating_weights = nn.functional.softmax(gating_weights, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # make the gating weights have the same shape as expert_outputs
        gating_weights = (
            gating_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        )

        return torch.sum(gating_weights * expert_outputs, dim=1)


class MOEResidualUnit(ResidualUnit):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        gate_input_dim: int,
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
    ):
        super().__init__(
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

        self.conv = MOESequential()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            if su == (subunits - 1):
                unit = MoEBlock(
                    self.spatial_dims,
                    schannels,
                    out_channels,
                    strides=sstrides,
                    kernel_size=kernel_size,
                    adn_ordering=adn_ordering,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    dropout_dim=dropout_dim,
                    dilation=dilation,
                    bias=bias,
                    conv_only=conv_only,
                    padding=padding,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                )
            else:
                unit = MOEConvolution(
                    self.spatial_dims,
                    schannels,
                    out_channels,
                    strides=sstrides,
                    kernel_size=kernel_size,
                    adn_ordering=adn_ordering,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    dropout_dim=dropout_dim,
                    dilation=dilation,
                    bias=bias,
                    conv_only=conv_only,
                    padding=padding,
                )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

    def forward(self, x, context):
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x, context)  # apply x to sequence of operations

        cx[:, : res.shape[1]] = (
            cx[:, : res.shape[1]] + res
        )  # add the residual to the output
        return cx


class MOEResUNetWithBottleneck(nn.Module):
    """
    A Residual U-Net implementation with an encoder-decoder structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        last_layer_conv_only: bool = True,
        num_experts: int = 4,
        gate_input_dim: int = 10,
    ) -> None:
        super().__init__()
        self.dropout = 0.2

        # Encoder
        self.encoder = nn.ModuleList(
            [
                self._create_residual_block(
                    in_channels,
                    16,
                    strides=2,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_residual_block(
                    16,
                    32,
                    strides=2,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_residual_block(
                    32,
                    64,
                    strides=2,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_residual_block(
                    64,
                    128,
                    strides=2,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_residual_block(
                    128,
                    256,
                    strides=1,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
            ]
        )

        # Decoder (Upsampling stages)
        self.decoder = nn.ModuleList(
            [
                self._create_upsample_block(
                    256 + 128,
                    64,
                    384,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_upsample_block(
                    64 + 64,
                    32,
                    128,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_upsample_block(
                    32 + 32,
                    16,
                    64,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
                self._create_upsample_block(
                    16 + 16,
                    out_channels,
                    32,
                    conv_only=last_layer_conv_only,
                    num_experts=num_experts,
                    gate_input_dim=gate_input_dim,
                ),
            ]
        )

        self.apply(self.initialize_weights)

    def _create_residual_block(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        num_experts: int,
        gate_input_dim: int,
    ) -> nn.Module:
        """
        Create a Residual Unit block with defined parameters.
        """
        return MOEResidualUnit(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=3,
            subunits=2,
            dropout=self.dropout,
            num_experts=num_experts,
            gate_input_dim=gate_input_dim,
        )

    def _create_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        num_experts: int = 4,
        gate_input_dim: int = 10,
        conv_only: bool = False,
    ) -> nn.Module:
        """
        Create an upsampling stage with a middle convolutional layer.
        """
        upsample = MOEUpsample(scale_factor=2, mode="trilinear", align_corners=False)
        up_conv_a = MOEConvolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            kernel_size=3,
            dropout=self.dropout,
        )
        if conv_only:
            up_conv_b = MOEConvolution(
                spatial_dims=3,
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                dropout=self.dropout,
                conv_only=conv_only,
            )
        else:
            up_conv_b = MoEBlock(
                spatial_dims=3,
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                dropout=self.dropout,
                num_experts=num_experts,
                gate_input_dim=gate_input_dim,
            )
        return MOESequential(upsample, up_conv_a, up_conv_b)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, give_feature: bool = False
    ) -> torch.Tensor:
        # Encoder pass
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x, context)
            encoder_outputs.append(x)

        if give_feature:
            bottleneck_feature = encoder_outputs[-1]

        # Start decoding from the last layer
        for i, decoder_stage in enumerate(self.decoder):
            x = torch.cat((x, encoder_outputs[-(i + 2)]), dim=1)
            x = decoder_stage(x, context)

        if give_feature:
            return x, bottleneck_feature
        else:
            return x

    @staticmethod
    def initialize_weights(module):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    x = torch.randn(2, 4, 128, 128, 128).cuda()
    context = torch.randn(2, 10).cuda()
    model = MOEResUNetWithBottleneck(4, 1).cuda()
    summary(model, input_data=(x, context))
    out = model(x, context)
    print(out.shape)
