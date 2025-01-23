import torch
import torch.nn as nn

from nets.residual_block import ResidualUnit_changed as ResidualUnit
from monai.networks.blocks import Convolution


class ResUNetCompact(nn.Module):
    """
    A Residual U-Net implementation with an encoder-decoder structure.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 1, last_layer_conv_only: bool = True
    ) -> None:
        super().__init__()
        self.dropout = 0.2

        # Encoder
        self.encoder = nn.ModuleList(
            [
                self._create_residual_block(in_channels, 16, strides=2),
                self._create_residual_block(16, 32, strides=2),
                self._create_residual_block(32, 64, strides=2),
                self._create_residual_block(64, 128, strides=2),
                self._create_residual_block(128, 256, strides=1),
            ]
        )

        # Decoder (Upsampling stages)
        self.decoder = nn.ModuleList(
            [
                self._create_upsample_block(256 + 128, 64, 384),
                self._create_upsample_block(64 + 64, 32, 128),
                self._create_upsample_block(32 + 32, 16, 64),
                self._create_upsample_block(
                    16 + 16, out_channels, 32, last_layer_conv_only
                ),
            ]
        )

    def _create_residual_block(
        self, in_channels: int, out_channels: int, strides: int
    ) -> nn.Module:
        """
        Create a Residual Unit block with defined parameters.
        """
        return ResidualUnit(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=3,
            subunits=2,
            dropout=self.dropout,
        )

    def _create_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        conv_only: bool = False,
    ) -> nn.Module:
        """
        Create an upsampling stage with a middle convolutional layer.
        """
        upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        up_conv_a = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            kernel_size=3,
            dropout=self.dropout,
        )
        up_conv_b = Convolution(
            spatial_dims=3,
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            dropout=self.dropout,
            conv_only=conv_only,
        )
        return nn.Sequential(upsample, up_conv_a, up_conv_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)

        # Start decoding from the last layer
        for i, decoder_stage in enumerate(self.decoder):
            x = torch.cat((x, encoder_outputs[-(i + 2)]), dim=1)
            x = decoder_stage(x)

        return x


class ResUNetWithBottleneck(nn.Module):
    """
    A Residual U-Net implementation with an encoder-decoder structure.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 1, last_layer_conv_only: bool = True
    ) -> None:
        super().__init__()
        self.dropout = 0.2

        # Encoder
        self.encoder = nn.ModuleList(
            [
                self._create_residual_block(in_channels, 16, strides=2),
                self._create_residual_block(16, 32, strides=2),
                self._create_residual_block(32, 64, strides=2),
                self._create_residual_block(64, 128, strides=2),
                self._create_residual_block(128, 256, strides=1),
            ]
        )

        # Decoder (Upsampling stages)
        self.decoder = nn.ModuleList(
            [
                self._create_upsample_block(256 + 128, 64, 384),
                self._create_upsample_block(64 + 64, 32, 128),
                self._create_upsample_block(32 + 32, 16, 64),
                self._create_upsample_block(
                    16 + 16, out_channels, 32, last_layer_conv_only
                ),
            ]
        )

        self.apply(self.initialize_weights)

    def _create_residual_block(
        self, in_channels: int, out_channels: int, strides: int
    ) -> nn.Module:
        """
        Create a Residual Unit block with defined parameters.
        """
        return ResidualUnit(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=3,
            subunits=2,
            dropout=self.dropout,
        )

    def _create_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        conv_only: bool = False,
    ) -> nn.Module:
        """
        Create an upsampling stage with a middle convolutional layer.
        """
        upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        up_conv_a = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            kernel_size=3,
            dropout=self.dropout,
        )
        up_conv_b = Convolution(
            spatial_dims=3,
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            dropout=self.dropout,
            conv_only=conv_only,
        )
        return nn.Sequential(upsample, up_conv_a, up_conv_b)

    def forward(self, x: torch.Tensor, give_feature: bool = False) -> torch.Tensor:
        # Encoder pass
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)

        if give_feature:
            bottleneck_feature = encoder_outputs[-1]

        # Start decoding from the last layer
        for i, decoder_stage in enumerate(self.decoder):
            x = torch.cat((x, encoder_outputs[-(i + 2)]), dim=1)
            x = decoder_stage(x)

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
    model = ResUNetCompact(4, 1).cuda()
    x = torch.randn((1, 4, 128, 128, 128)).cuda()
    y = model(x)
    print(y.shape)
    # model = ResUNetWithBottleneck(1, 1)
    # x = torch.randn((1, 1, 128, 128, 128))
    # y = model(x)
    # print(y.shape)
    # y = model(x, give_feature=True)
    # print(y[0].shape, y[1].shape)
