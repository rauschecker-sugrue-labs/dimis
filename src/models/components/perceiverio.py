from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from perceiver.model.core import (
    FourierPositionEncoding,
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    TrainableQueryProvider,
)


class DomainInputAdapter(InputAdapter):
    """An InputAdapter which can handle data in k-space domain or pixel domain.

    Transforms and position-encodes task-specific input to generic encoder
    input of shape (B, M, C) where B is the batch size, M the input sequence
    length and C the number of key/value input channels. C is determined by the
    `num_input_channels` property of the `input_adapter`. In this case,
    x-slice dimension acts as input channels.
    """

    def __init__(
        self, input_shape: Tuple[int], num_frequency_bands: int
    ) -> None:
        """Initialization of the domain input adapter for the PerceiverIO.

        Args:
            input_shape: Shape of the input.
            num_frequency_bands: Number of frequency bands for the positional
                encoding.
        """
        self.input_shape = input_shape
        spatial_shape = [*input_shape[:2], *input_shape[3:]]
        num_slices = input_shape[2]

        if num_frequency_bands == 0:
            super().__init__(num_input_channels=num_slices)
            self.position_encoding = None
        else:
            position_encoding = FourierPositionEncoding(
                input_shape=spatial_shape,
                num_frequency_bands=num_frequency_bands,
            )

            super().__init__(
                num_input_channels=num_slices
                + position_encoding.num_position_encoding_channels()
            )
            self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the initialized positional encoding with the input.

        Args:
            x: Input.

        Raises:
            ValueError: If positional encoding does not have the same shape as
                the input.

        Returns:
            Concatenated input and positional encoding.
        """
        b, *d = x.shape

        # Check if the given input shape and positional encoding are compatible
        if tuple(d) != self.input_shape:
            raise ValueError(
                f'Input shape {tuple(d)} different from \
                required shape {self.input_shape}'
            )

        # b (batch), c (class), v (real/imag-part), x, y, z
        x = rearrange(x, 'b c v x y z -> b (c v z y) x')
        if self.position_encoding is None:
            return x

        x_enc = self.position_encoding(b)
        return torch.cat([x, x_enc], dim=-1)


class SegmentationOutputAdapter(OutputAdapter):
    """Transforms generic decoder cross-attention output to segmentation map."""

    def __init__(self, output_len: int, num_output_query_channels: int) -> None:
        """Initialization of the segmentation output adapter.

        Args:
            output_len: Desired length of the output.
            num_output_query_channels: Desired number of output query channels.
        """
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies linear layer to the input to generate desired output shape.

        Args:
            x: Input.
        Returns:
            Output in desired shape.
        """
        return self.linear(x)


class PerceiverIO(nn.Module):
    """Implementation of PerceiverIO
        (See https://github.com/krasserm/perceiver-io).

    The model uses a specified encoder and decoder.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        num_frequency_bands: int,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int,
        num_cross_attention_layers: int,
        num_self_attention_heads: int,
        num_self_attention_layers_per_block: int,
        num_self_attention_blocks: int,
        dropout: float,
    ) -> None:
        """Initialization of the PerceiverIO model.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            num_frequency_bands: Number of frequency bands used for positional
                encoding.
            num_latents: Number of latent values.
            num_latent_channels: Number of latent channels.
            num_cross_attention_heads: Number of cross-attention heads.
            num_cross_attention_layers: Number of cross-attention layers.
            num_self_attention_heads: Number of self-attention heads.
            num_self_attention_layers_per_block: Number of self-attention
                layers per block.
            num_self_attention_blocks: Number of self-attention blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        output_len = int(np.prod(output_shape))
        output_query_channels = 1
        input_adapter = DomainInputAdapter(
            self.input_shape,
            num_frequency_bands,
        )
        output_query_provider = TrainableQueryProvider(
            num_queries=1,  # Number of output channels
            num_query_channels=output_query_channels,
            init_scale=0.02,  # scale for Gaussian query initialization
        )
        output_adapter = SegmentationOutputAdapter(
            output_len, output_query_channels
        )
        modules = (
            PerceiverEncoder(
                input_adapter=input_adapter,
                num_latents=num_latents,  # N
                num_latent_channels=num_latent_channels,  # D
                num_cross_attention_heads=num_cross_attention_heads,
                num_cross_attention_layers=num_cross_attention_layers,
                first_cross_attention_layer_shared=False,
                num_self_attention_heads=num_self_attention_heads,
                num_self_attention_layers_per_block=(
                    num_self_attention_layers_per_block
                ),
                num_self_attention_blocks=num_self_attention_blocks,
                first_self_attention_block_shared=True,
                dropout=dropout,
                init_scale=0.02,  # scale for Gaussian latent initialization
                activation_checkpointing=False,
                activation_offloading=False,
            ),
            PerceiverDecoder(
                output_adapter=output_adapter,
                output_query_provider=output_query_provider,
                num_latent_channels=num_latent_channels,
                num_cross_attention_qk_channels=4,
            ),
        )
        self.linear_nonlinear = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the PerceiverIO.

        Args:
            x: Input.

        Returns:
            Predicted values of shape (b, c, v, x, y, z).
        """
        x = self.linear_nonlinear(x)
        # b (batch), q (query), c (class), v (real/imag-part), x, y, z
        return rearrange(
            x,
            'b q (c v x y z) -> (b q) c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            x=self.output_shape[2],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )


if __name__ == "__main__":
    _ = PerceiverIO()
