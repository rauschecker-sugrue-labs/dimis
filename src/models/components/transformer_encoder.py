from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.perceiverio import DomainInputAdapter


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
        heads: int = 2,
        num_frequency_bands: int = 0,
        dropout: float = 0.2,
    ) -> None:
        """Initialization of the transformer encoder.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of transformer encoder layers. Defaults to 1.
        """
        super().__init__()
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        hidden_size = int(input_len * hidden_factor)
        self.input_adapter = DomainInputAdapter(
            input_shape, num_frequency_bands
        )

        in_linear = nn.Linear(input_len, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size, dropout
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layers, depth)
        out_linear = nn.Linear(hidden_size, output_len)

        self.layers = nn.ModuleList(
            [in_linear, transformer_encoder, out_linear]
        )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the network.

        Args:
            x: Input of shape (b c v x y z).

        Returns:
            Output of the network.
        """
        x = self.input_adapter(x)
        # Input adapter is rearranging in a d different order so correct it
        x = rearrange(x, 'b l x -> b x l')
        x = self.layers(x)
        return rearrange(
            x,
            'b x (c v y z) -> b c v x y z',
            c=self.output_shape[0],
            v=self.output_shape[1],
            y=self.output_shape[3],
            z=self.output_shape[4],
        )
    
    if __name__ == "__main__":
    _ = Transformer()
