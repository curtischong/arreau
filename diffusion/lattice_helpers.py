import torch


# we are using this better way to encode angles: https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network
def encode_angles(angles: torch.Tensor) -> torch.Tensor:
    # I verified in a debugger that these are the same:
    # torch.atan2(torch.sin(angles), torch.cos(angles)) == angles
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def decode_angles(angles: torch.Tensor) -> torch.Tensor:
    return torch.atan2(angles[:, :3], angles[:, 3:])


# https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L67
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
    # angles = angles * 180.0 / torch.pi # convert radians to degrees
    return torch.cat([lengths, angles], dim=1)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return torch.clamp(val, -max_abs_val, max_abs_val)


# https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
def lattice_from_params(
    params: torch.Tensor,
):
    """
    Create a Lattice using unit cell lengths and angles (in degrees).

    Args:
        a (float): *a* lattice parameter.
        b (float): *b* lattice parameter.
        c (float): *c* lattice parameter.
        alpha (float): *alpha* angle in degrees.
        beta (float): *beta* angle in degrees.
        gamma (float): *gamma* angle in degrees.

    Returns:
        Lattice with the specified lattice parameters.
    """
    a, b, c, alpha, beta, gamma = params.unbind(-1)
    # beta = torch.deg2rad(beta)
    # gamma = torch.deg2rad(gamma)
    # alpha = torch.deg2rad(alpha)
    num_lattices = a.shape[0]

    # angles_r = torch.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_alpha, sin_beta = torch.sin(alpha), torch.sin(beta)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [a * sin_beta, torch.zeros(num_lattices), a * cos_beta], dim=1
    )
    vector_b = torch.stack(
        [
            -b * sin_alpha * torch.cos(gamma_star),
            b * sin_alpha * torch.sin(gamma_star),
            b * cos_alpha,
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [torch.zeros(num_lattices), torch.zeros(num_lattices), c], dim=1
    )

    res = torch.cat([vector_a, vector_b, vector_c], dim=-1).view(num_lattices, 3, 3)
    return res


# def lattice_params_to_matrix_torch(lattice_parameters: torch.Tensor):
#     """Batched torch version to compute lattice matrix from params.

#     lengths: torch.Tensor of shape (N, 3), unit A
#     angles: torch.Tensor of shape (N, 3), unit degree
#     """
#     lengths = lattice_parameters[:, :3]
#     angles = lattice_parameters[:, 3:]
#     angles_r = torch.deg2rad(angles)
#     coses = torch.cos(angles_r)
#     sins = torch.sin(angles_r)

#     val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
#     # Sometimes rounding errors result in values slightly > 1.
#     val = torch.clamp(val, -1.0, 1.0)
#     gamma_star = torch.arccos(val)

#     vector_a = torch.stack(
#         [
#             lengths[:, 0] * sins[:, 1],
#             torch.zeros(lengths.size(0), device=lengths.device),
#             lengths[:, 0] * coses[:, 1],
#         ],
#         dim=1,
#     )
#     vector_b = torch.stack(
#         [
#             -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
#             lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
#             lengths[:, 1] * coses[:, 0],
#         ],
#         dim=1,
#     )
#     vector_c = torch.stack(
#         [
#             torch.zeros(lengths.size(0), device=lengths.device),
#             torch.zeros(lengths.size(0), device=lengths.device),
#             lengths[:, 2],
#         ],
#         dim=1,
#     )

#     return torch.stack([vector_a, vector_b, vector_c], dim=1)
