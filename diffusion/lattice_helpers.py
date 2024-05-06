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
    return lengths, angles


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

    we expect params to be: [length1, length2, length3, alpha, beta, gamma]
    where alpha, beta, gamma are in radians


    Returns:
        Lattice with the specified lattice parameters.
    """
    a, b, c, alpha, beta, gamma = params.unbind(-1)
    # beta = torch.deg2rad(beta)
    # gamma = torch.deg2rad(gamma)
    # alpha = torch.deg2rad(alpha)
    num_lattices = a.shape[0]

    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_alpha, sin_beta = torch.sin(alpha), torch.sin(beta)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [a * sin_beta, torch.zeros(num_lattices, device=a.device), a * cos_beta], dim=1
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
        [
            torch.zeros(num_lattices, device=a.device),
            torch.zeros(num_lattices, device=a.device),
            c,
        ],
        dim=1,
    )

    res = torch.cat([vector_a, vector_b, vector_c], dim=-1).view(num_lattices, 3, 3)
    return res
