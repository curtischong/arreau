import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from ponita.utils.to_from_sphere import sphere_to_scalar, sphere_to_vec
from ponita.nn.embedding import PolynomialFeatures
from ponita.utils.windowing import PolynomialCutoff
from ponita.transforms import PositionOrientationGraph, SEnInvariantAttributes
from torch_geometric.transforms import Compose
from torch_scatter import scatter_add, scatter_mean
from ponita.nn.conv import Conv, FiberBundleConv
from ponita.nn.convnext import ConvNext
from torch_geometric.transforms import BaseTransform, Compose, RadiusGraph


# Wrapper to automatically switch between point cloud mode (num_ori = -1 or 0) and
# bundle mode (num_ori > 0).
def Ponita(input_dim, hidden_dim, output_dim, num_layers, output_dim_vec = 0, radius = None,
           num_ori=20, basis_dim=None, degree=3, widening_factor=4, layer_scale=None,
           task_level='graph', multiple_readouts=True, lift_graph=False, **kwargs):
    # Select either FiberBundle mode or PointCloud mode
    PonitaClass = PonitaFiberBundle if (num_ori > 0) else PonitaPointCloud
    # Return the ponita object
    return PonitaClass(input_dim, hidden_dim, output_dim, num_layers, output_dim_vec = output_dim_vec, 
                       radius = radius, num_ori=num_ori, basis_dim=basis_dim, degree=degree, 
                       widening_factor=widening_factor, layer_scale=layer_scale, task_level=task_level, 
                       multiple_readouts=multiple_readouts, lift_graph=lift_graph, **kwargs)


class PonitaFiberBundle(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 output_dim_global_scalar: int,
                 output_dim_global_vec: int,
                 num_layers,
                 output_dim_vec = 0,
                 radius = None,
                 num_ori=20,
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 multiple_readouts=True,
                 **kwargs):
        super().__init__()

        # Input output settings
        self.output_dim, self.output_dim_vec, self.output_dim_global_scalar, self.output_dim_global_vec = output_dim, output_dim_vec, output_dim_global_scalar, output_dim_global_vec

        # For constructing the position-orientation graph and its invariants
        self.transform = Compose([PositionOrientationGraph(num_ori), SEnInvariantAttributes(separable=True)])

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(PolynomialFeatures(degree), nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        self.fiber_basis_fn = nn.Sequential(PolynomialFeatures(degree), nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        self.windowing_fn = PolynomialCutoff(radius)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False)
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = FiberBundleConv(hidden_dim, hidden_dim, basis_dim, groups=hidden_dim, separable=True)
            layer = ConvNext(hidden_dim, conv, act=act_fn, layer_scale=layer_scale, widening_factor=widening_factor)
            self.interaction_layers.append(layer)
            # self.interaction_layers.append(ConvNextR3S2(hidden_dim, basis_dim, act=act_fn, widening_factor=widening_factor, layer_scale=layer_scale))
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(nn.Linear(hidden_dim, output_dim + output_dim_vec + output_dim_global_scalar + output_dim_global_vec))
            else:
                self.read_out_layers.append(None)
    
    def forward(self, graph):

        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        # PERF: do we need to use this windowing function? Since the dists are already masked to be within the cutoff radius
        kernel_basis = self.basis_fn(graph.attr) * torch.unsqueeze(self.windowing_fn(graph.dists), (-1, -1))
        fiber_kernel_basis = self.fiber_basis_fn(graph.fiber_attr)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, fiber_attr=fiber_kernel_basis, batch=graph.batch)
            if readout_layer is not None: readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)
        
        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec, readout_global_vec, readout_global_scalar = torch.split(readout, [self.output_dim, self.output_dim_vec, self.output_dim_global_vec, self.output_dim_global_scalar], dim=-1)
        
        # Read out scalar and vector predictions
        output_scalar = self.scalar_readout_fn(readout_scalar)
        output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid)
        global_output_scalar = self.global_scalar_readout_fn(readout_global_scalar, graph.batch)
        global_output_vector = self.global_vec_readout_fn(readout_global_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar, output_vector, global_output_scalar, global_output_vector
    
    def scalar_readout_fn(self, readout_scalar):
        if self.output_dim > 0:
            output_scalar = sphere_to_scalar(readout_scalar)
        else:
            output_scalar = None
        return output_scalar
    
    def vec_readout_fn(self, readout_vec, ori_grid):
        if self.output_dim_vec > 0:
            output_vector = sphere_to_vec(readout_vec, ori_grid)
        else:
            output_vector = None
        return output_vector
    
    def global_vec_readout_fn(self, readout_vec, ori_grid, batch):
        if self.output_dim_global_vec > 0:
            output_vectors = sphere_to_vec(readout_vec, ori_grid)
            # we are not using global_add_pool since it doesn't work with dim()=3 tensors (it'll sum on the wrong dimension)
            output_vector = scatter_add(output_vectors, batch, dim=0) # sum over the batch dimension (dim=0) but group by the batch index
        else:
            output_vector = None
        return output_vector
    
    def global_scalar_readout_fn(self, readout_scalar, batch):
        if self.output_dim_global_scalar > 0:
            output_scalar = sphere_to_scalar(readout_scalar)
            output_scalar=global_add_pool(output_scalar, batch)
        else:
            output_scalar = None
        return output_scalar


class PonitaPointCloud(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 output_dim_vec = 0,
                 radius = None,
                 num_ori = -1,
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 task_level='graph',
                 multiple_readouts=True, 
                 lift_graph=False,
                 **kwargs):
        super().__init__()

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = (task_level=='graph') # TODO: remove this. it's not necessary. we will always do some global pooling, some not

        # For constructing the position-orientation graph and its invariants
        self.lift_graph = lift_graph
        if lift_graph:
            self.transform = Compose([PositionOrientationGraph(num_ori, radius), SEnInvariantAttributes(separable=False, point_cloud=True)])

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(PolynomialFeatures(degree), nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        self.windowing_fn = PolynomialCutoff(radius)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False)
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = Conv(hidden_dim, hidden_dim, basis_dim, groups=hidden_dim)
            layer = ConvNext(hidden_dim, conv, act=act_fn, layer_scale=layer_scale, widening_factor=widening_factor)
            self.interaction_layers.append(layer)
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(nn.Linear(hidden_dim, output_dim + output_dim_vec))
            else:
                self.read_out_layers.append(None)
    
    def forward(self, graph):

        # Lift and compute invariants
        if self.lift_graph:
            graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, batch=graph.batch)
            if readout_layer is not None: readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)
        
        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)
        
        # Read out scalar and vector predictions (if pos-ori cloud collect all predictions that have the same base point in R^n)
        if hasattr(graph, 'scatter_projection_index'):
            output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch, graph.scatter_projection_index)
            output_vector = self.vec_readout_fn(readout_vec, graph.pos, graph.batch, graph.scatter_projection_index)
        else:
            output_scalar = readout_scalar
            if self.global_pooling:
                output_scalar=global_add_pool(output_scalar, graph.batch)
            output_vector = None

        # Return predictions
        return output_scalar, output_vector
    
    def scalar_readout_fn(self, readout_scalar, batch, scatter_projection_index):
        if self.output_dim > 0:
            # Aggregate predictions toward the base position in R^n
            output_scalar = scatter_mean(readout_scalar, scatter_projection_index, dim=0)
            if self.global_pooling:
                batch_Rn = scatter_mean(batch, scatter_projection_index, dim=0).type_as(batch)
                output_scalar=global_add_pool(output_scalar, batch_Rn)
        else:
            output_scalar = None
        return output_scalar
    
    def vec_readout_fn(self, readout_vec, pos, batch, scatter_projection_index):
        if self.output_dim_vec > 0:
            # Scale each orientation with the predicted scalar and aggregate via scatter_mean
            _, ori = pos.split(int(pos.shape[-1]/2),dim=-1)
            output_vector = scatter_mean(readout_vec[:,:,None] * ori[:,None,:], scatter_projection_index, dim=0)   
            if self.global_pooling:
                batch_Rn = scatter_mean(batch, scatter_projection_index, dim=0).type_as(batch)
                output_vector = global_add_pool(output_vector, batch_Rn)
        else:
            output_vector = None
        return output_vector
    