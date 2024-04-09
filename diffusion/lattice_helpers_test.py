import torch

from lattice_helpers import lattice_from_params, matrix_to_params


def main():
    lattice = torch.tensor(
        [
            [
                [5.28526086, 0.0, 0.0],
                [2.64263043, 4.57717017, 0.0],
                [2.64263043, 1.52572339, 4.31539742],
            ],
            [
                [5.52431857, 0.0, 0.0],
                [2.76215929, 4.78420022, 0.0],
                [2.76215929, 1.59473341, 4.51058723],
            ],
        ]
    )
    print(lattice)
    params = matrix_to_params(lattice)
    print(params)
    lattice = lattice_from_params(params)
    print(lattice)
    params = matrix_to_params(lattice)
    print(params)
    lattice = lattice_from_params(params)
    print(lattice)
    params = matrix_to_params(lattice)
    print(params)
    lattice = lattice_from_params(params)
    # lattice = get_lattice_parameters(params)
    print(lattice)

    # basically, there are different representations of the same lattice. we just need to make sure that the code is consistent with the same form
    # these two functions encode/decode the same lattice


if __name__ == "__main__":
    main()
