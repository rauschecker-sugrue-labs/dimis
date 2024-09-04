import torch


def min_max_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Scales a tensor between 0 and 1 on spatial dimensions.

    Args:
        tensor: Tensor of shape (batch, class, channel, x, y, z).

    Returns:
        Scaled tensor.
    """
    assert (
        len(tensor.shape) == 6
    ), f'Tensor rank mismatch: expected a rank of 6, got {len(tensor.shape)}'

    min_vals = torch.amin(tensor, dim=(-3, -2, -1), keepdim=True)
    max_vals = torch.amax(tensor, dim=(-3, -2, -1), keepdim=True)

    return (tensor - min_vals) / (max_vals - min_vals)
