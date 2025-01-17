from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class MLP_Block(nn.Module):
    """Building block for MLP-based models."""

    def __init__(
        self, hidden_size: int, activation: nn.Module, depth: int
    ) -> None:
        """Initialization of the MLP block.

        Args:
            hidden_size: Number of neurons in the linear layer.
            activation: Activation function.
            depth: Number of MLP blocks (linear layer with activation).
        """
        super(MLP_Block, self).__init__()
        layers = []
        for _ in range(depth):
            linear = nn.Linear(hidden_size, hidden_size)
            layers.append(linear)
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Propagates the input through the MLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
    ) -> None:
        """Initialization of the multi-layer perceptron.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of hidden layers. Defaults to 1.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_len, hidden_size),  # Input layer
                MLP_Block(hidden_size, nn.Tanh(), depth),
                nn.Linear(hidden_size, output_len),  # Output layer
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the network.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        # Rearrange to let the net process the input sagittally
        x = rearrange(x, 'b c v x y z -> b x (c v y z)')
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
    _ = MLP()
