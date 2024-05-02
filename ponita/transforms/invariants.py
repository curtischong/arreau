import torch
from torch_geometric.transforms import BaseTransform
from ponita.geometry.invariants import invariant_attr_r2s1_fiber_bundle, invariant_attr_r2s1_point_cloud
from ponita.geometry.invariants import invariant_attr_rn, invariant_attr_r3s2_fiber_bundle, invariant_attr_r3s2_point_cloud
from ponita.utils.to_from_sphere import scalar_to_sphere


class SEnInvariantAttributes(BaseTransform):
    """
    A PyTorch Geometric transform that adds invariant edge attributes to the input graph.
    The transformation includes pair-wise distances in position space (graph.dists) and
    invariant edge attributes between local orientations.

    Args:
        separable (bool): If True, computes spatial invariants for each orientation separately
                         (no orientation interactions). If False, computes all pair-wise
                         invariants between orientations in the receiving fiber related to
                         those in the sending fiber.
    """

    def __init__(self, separable=True, point_cloud=False):
        super().__init__()
        # Discretization of the orientation grid
        self.separable = separable
        self.point_cloud = point_cloud
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def __call__(self, graph):
        """
        Apply the transform to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                orientation (graph.ori), and edge index (graph.edge_index).

        Returns:
            torch_geometric.data.Data: Updated graph with added invariant edge attributes.
                                      Pair-wise distances in position space are stored in graph.dists.
                                      If separable is True, graph.attr contains spatial invariants for
                                      each orientation, and graph.attr_ori contains invariants between
                                      local orientations. If separable is False, graph.attr contains
                                      all pair-wise invariants between orientations.
        """
        # TODO: make more elegant
        if graph.n == 2:
            graph.dists = invariant_attr_rn(graph.pos[:,:graph.n], graph.edge_index)
            if self.point_cloud:
                if graph.pos.size(-1) == graph.n: 
                    graph.attr = graph.dists
                else:
                    graph.attr = invariant_attr_r2s1_point_cloud(graph.pos, graph.edge_index)
                return graph
            else:
                if self.separable:
                    graph.attr, graph.fiber_attr = invariant_attr_r2s1_fiber_bundle(graph.pos, graph.ori_grid, graph.edge_index, separable=True)
                else:
                    graph.attr = invariant_attr_r2s1_fiber_bundle(graph.pos, graph.ori_grid, graph.edge_index, separable=False)
                return graph
        else:
            # assume that graph.dists is already computed (when we calculated the edge_index)
            # graph.dists = invariant_attr_rn(graph.pos[:,:graph.n], graph.edge_index)
            graph.attr = graph.dists
            if self.point_cloud:
                if graph.pos.size(-1) == graph.n: 
                    graph.attr = graph.dists
                else:
                    graph.attr = invariant_attr_r3s2_point_cloud(graph.pos, graph.edge_index)
                return graph
            else:
                if self.separable:
                    # TODO: I'm assuming we do NOT need to add the features to the fiber_attr???
                    r3s2_attr, graph.fiber_attr = invariant_attr_r3s2_fiber_bundle(graph.pos, graph.ori_grid, graph.edge_index, graph.inter_atom_direction, separable=True)

                    # Since the other features that I'm adding are directionless scalars, it's similar to a norm. So I think
                    # I can just concat them to the end of the attributes
                    # I came to this conclusion after looking at Theorem 1 of the ponita paper


                    # Note: I believe that adding graph.dists is a useful feature since it will help the model determine by how much it should scale this edge (to determine the lattice lengths)                    
                    # The cosine similarities are for equation (A39) in mattergen
                    lattice_for_edge = torch.index_select(graph.lattice, 0, graph.batch_of_edge)
                    angle_diff_0 = self.cosine_similarity(graph.inter_atom_direction, lattice_for_edge[:, 0, :]) # note: cos is an even function, so the order we subtract doesn't matter
                    angle_diff_1 = self.cosine_similarity(graph.inter_atom_direction, lattice_for_edge[:, 1, :])
                    angle_diff_2 = self.cosine_similarity(graph.inter_atom_direction, lattice_for_edge[:, 2, :])
                    graph.edge_scalar_features = scalar_to_sphere(torch.stack([graph.dists, angle_diff_0, angle_diff_1, angle_diff_2], dim=-1), graph.ori_grid)

                    graph.attr = torch.cat([r3s2_attr, graph.edge_scalar_features], dim=-1)

                else:
                    graph.attr = invariant_attr_r3s2_fiber_bundle(graph.pos, graph.ori_grid, graph.edge_index, separable=False)
                return graph