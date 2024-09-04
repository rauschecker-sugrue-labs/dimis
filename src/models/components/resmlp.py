import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.mlp import MLP_Block


class Affine(nn.Module):
    """Affine Layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class ResMLP_Block(nn.Module):
    """ResMLP building block."""

    def __init__(
        self, num_patches: tuple[int], latent_dim: int, layerscale_init: float
    ):
        """Building block for the ResMLP.

        Args:
            num_patches: Number of patches.
            latent_dim: Length of the latent dimension.
            layerscale_init: Layerscale initialization.
        """
        super().__init__()
        self.affine_1 = Affine(latent_dim)
        self.affine_2 = Affine(latent_dim)
        self.linear_patches = nn.Linear(num_patches, num_patches)
        self.mlp_channels = MLP_Block(latent_dim, nn.GELU(), 1)

        self.layerscale_1 = nn.Parameter(
            layerscale_init * torch.ones((latent_dim))
        )  # LayerScale parameters
        self.layerscale_2 = nn.Parameter(
            layerscale_init * torch.ones((latent_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the input through the ResMLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        res_1 = rearrange(
            self.linear_patches(rearrange(self.affine_1(x), 'b x l -> b l x')),
            'b l x -> b x l',
        )
        x = x + self.layerscale_1 * res_1
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        return x


class ResMLP(nn.Module):
    """ResMLP model: Stacking the full network.
    (See https://arxiv.org/pdf/2105.03404.pdf)"""

    def __init__(
        self,
        input_shape: tuple[int],
        output_shape: tuple[int],
        hidden_factor: int = 1,
        depth: int = 1,
        layerscale_init: float = 0.2,
    ):
        """Initialization of the ResMLP model.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of ResMLP blocks. Defaults to 1.
            layerscale_init: Layerscale for the normalization. Defaults to 0.2.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod([*input_shape[:2], *input_shape[3:]]))
        output_len = int(np.prod([*output_shape[:2], *output_shape[3:]]))
        num_patches = input_shape[2]
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [nn.Linear(input_len, hidden_size)]
            + [
                ResMLP_Block(
                    num_patches,
                    hidden_size,
                    layerscale_init,
                )
                for i in range(depth)
            ]
            + [
                Affine(hidden_size),
                nn.Linear(hidden_size, output_len),
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
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
    _ = ResMLP()
