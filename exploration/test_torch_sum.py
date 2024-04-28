# layer, batch, value dimension
# 3 layers, 2 batches, 4 values
import torch

res = torch.tensor(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)

print(torch.sum(res, dim=0))

# output:
# tensor([[27, 30, 33, 36],
#         [39, 42, 45, 48]])
# size 2x4 which is correct
