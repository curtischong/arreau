# https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L67
import torch


def matrix_to_params(matrix: torch.Tensor) -> torch.Tensor:
    """
    Returns the angles (alpha, beta, gamma) of the lattice.
    """
    m = matrix
    lengths = torch.sqrt(torch.sum(matrix**2, dim=-1))
    angles = torch.zeros((matrix.shape[0], 3), device=matrix.device)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.acos(
            torch.clamp(
                torch.sum(m[..., j, :] * m[..., k, :], dim=-1)
                / (lengths[..., j] * lengths[..., k]),
                -1.0,
                1.0,
            )
        )
    # angles = angles * 180.0 / torch.pi # scrw radians. we're using degrees
    return torch.cat([lengths, angles], dim=1)


def lattice_params_to_matrix_torch(lattice_parameters: torch.Tensor):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    lengths = lattice_parameters[:, :3]
    angles = lattice_parameters[:, 3:]
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)
