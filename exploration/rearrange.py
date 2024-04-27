import torch

coords = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)

coords2 = torch.tensor(
    [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
)

rearranged_coords2 = coords2.transpose(0, 1).reshape(-1, 3)


def assert_tensors_equal(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    assert torch.all(tensor1 == tensor2), "Tensors are not equal at all indices"


assert_tensors_equal(coords, rearranged_coords2)
