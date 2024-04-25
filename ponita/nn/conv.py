import torch
import torch_geometric
from typing import Any, Optional
from torch_geometric.typing import Adj, Size
from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)


class Conv(torch_geometric.nn.MessagePassing):
    """
    """
    def __init__(self, in_channels, out_channels, attr_dim, bias=True, aggr="add", groups=1):
        super().__init__(node_dim=0, aggr=aggr)
        
        # Check arguments
        if groups==1:
            self.depthwise = False
        elif groups==in_channels and groups==out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')
        
        # Construct kernel and bias
        self.kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        
    def forward(self, x, edge_index, edge_attr, **kwargs):
        """
        """
        # Sample the convolution kernels
        kernel = self.kernel(edge_attr)

        # Do the convolution
        out = self.propagate(edge_index, x=x, kernel=kernel)

        # Re-callibrate the initializaiton
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), out.std())

        # Add bias
        if self.bias is not None:
            return out + self.bias
        else:  
            return out

    def message(self, x_j, kernel):
        if self.depthwise:
            return kernel * x_j
        else:
            return torch.einsum('boi,bi->bo', kernel.unflatten(-1, (self.out_channels, self.in_channels)), x_j)
    
    def callibrate(self, std_in, std_out):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in/std_out
            self.callibrated = ~self.callibrated


class FiberBundleConv(torch_geometric.nn.MessagePassing):
    """
    """
    def __init__(self, in_channels, out_channels, attr_dim, bias=True, aggr="add", separable=True, groups=1):
        super().__init__(node_dim=0, aggr=aggr)

        # Check arguments
        if groups==1:
            self.depthwise = False
        elif groups==in_channels and groups==out_channels:
            self.depthwise = True
            self.in_channels = in_channels
            self.out_channels = out_channels
        else:
            assert ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')

        # Construct kernels
        self.separable = separable
        if self.separable:
            self.kernel = torch.nn.Linear(attr_dim, in_channels, bias=False)
            self.fiber_kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)
        else:
            self.kernel = torch.nn.Linear(attr_dim, int(in_channels * out_channels / groups), bias=False)
        
        # Construct bias
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)
        
        # Automatic re-initialization
        self.register_buffer("callibrated", torch.tensor(False))
        
    def forward(self, x, edge_index, edge_attr, fiber_attr=None, **kwargs):
        """
        """

        # Do the convolutions: 1. Spatial conv, 2. Spherical conv
        kernel = self.kernel(edge_attr)
        # edge_features = self.message(x, kernel)
        x_1 = self.propagate2(edge_index, x=x, kernel=kernel)
        if self.separable:
            fiber_kernel = self.fiber_kernel(fiber_attr)
            if self.depthwise:
                x_2 = torch.einsum('boc,opc->bpc', x_1, fiber_kernel) / fiber_kernel.shape[-2]
            else:
                x_2 = torch.einsum('boc,opdc->bpd', x_1, fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels))) / fiber_kernel.shape[-2]
        else:
            x_2 = x_1

        # Re-callibrate the initializaiton
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        else:  
            return x_2

    def message(self, x_j, kernel):
        if self.separable:
            return kernel * x_j
        else:
            if self.depthwise:
                return torch.einsum('bopc,boc->bpc', kernel, x_j)
            else:
                return torch.einsum('bopdc,boc->bpd', kernel.unflatten(-1, (self.out_channels, self.in_channels)), x_j)
    
    def callibrate(self, std_in, std_1, std_2):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel.weight.data = self.kernel.weight.data * std_in/std_1
            if self.separable:
                self.fiber_kernel.weight.data = self.fiber_kernel.weight.data * std_1/std_2
            self.callibrated = ~self.callibrated

    def propagate2(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if is_sparse(edge_index) and self.fuse and not self.explain:
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self._collect(self._user_args, edge_index,
                                          mutable_size, kwargs)

                msg_kwargs = self.inspector.collect_param_data(
                    'message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data(
                    'aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.collect_param_data(
                    'update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, mutable_size, kwargs), out)
            if res is not None:
                out = res

        return out