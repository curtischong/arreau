import torch
import torch_geometric

class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor # This is derived from the Xt and Lt properties. This is why we don't have the Xt and Lt fields below
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    A0: torch.Tensor | None # All of the "| None" types are because we don't have them during inference. However, the non-None types are needed to calculate the loss during training
    X0: torch.Tensor | None
    L0: torch.Tensor | None
    num_atoms: torch.Tensor

    def __init__(
        self,
        *,
        edge_index: torch.Tensor,  # [2, n_edges]
        # node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        A0: torch.Tensor | None, # the atomic number at time 0
        X0: torch.Tensor | None,
        L0: torch.Tensor | None,
    ):
        # Check shapes
        # num_nodes = node_attrs.shape[0]
        num_nodes = A0.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        # assert len(node_attrs.shape) == 2
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            # "node_attrs": node_attrs,
            "A0": A0,
            "X0": X0,
            "L0": L0,
            "num_atoms": A0.shape[0],
        }
        super().__init__(**data)
