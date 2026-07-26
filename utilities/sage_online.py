
from torch_geometric.nn.conv.message_passing import MessagePassing
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from typing import List, Optional, Tuple, Union, Any
from utilities.lstm import LSTMAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import is_sparse, spmm
from torch import Tensor
import torch
import inspect



class StatefulSAGEConv(MessagePassing):
    def __init__(self,
                in_channels: Union[int, Tuple[int, int]],
                out_channels: int,
                aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
                normalize: bool = False,
                root_weight: bool = True,
                project: bool = False,
                bias: bool = True,
                **kwargs):
        
        super().__init__(aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.saved_message = None
        self.lstm_aggr = LSTMAggregation(in_channels, in_channels)

        

        if self.project:
            if self.in_channels <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(self.in_channels, self.in_channels, bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                self.in_channels)
        else:
            aggr_out_channels = self.in_channels

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(self.in_channels, out_channels, bias=False)

        self.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:
        #print('SAGEConv forward')
        #print(x.shape)
        if isinstance(x, Tensor):
            x = (x, x)
        #print(x.shape)
        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate( x=x, edge_index=edge_index, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
    
    #def get_all_funcs(self, obj):
    #    """
    #    Returns a sorted list of all unique method names of an object,
    #    including inherited methods from its class hierarchy.
    #    """
    #    funcs = set()
    #    for cls in inspect.getmro(obj.__class__):
    #        for name, member in cls.__dict__.items():
    #            if callable(member):
    #                funcs.add(name)
    #    # Also check for instance-level methods (e.g., monkey-patched)
    #    for name in dir(obj):
    #        attr = getattr(obj, name)
    #        if callable(attr):
    #            funcs.add(name)
    #    return sorted(funcs)

    def propagate(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
        ) -> Tensor:
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

        #mutable_size = self._check_input(edge_index, size)
        mutable_size = [None, None]
        # Run "fused" message and aggregation (if applicable).
        fuse = False
        

        if fuse:
            print('combined')
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            #print('separate')
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

                msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
                out = self.message(**msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
                out = self.aggregate(out, **aggr_kwargs)

                update_kwargs = self.inspector.collect_param_data('update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)


        return out


    def message(self, x_j):
        # Save the message for reuse
        #self.saved_message = x_j
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # If a saved message exists, use it for aggregation
        #print(inputs.shape)
        #if self.saved_message is not None:
        #    inputs = self.saved_message
        #print('here')
        aggregated = self.lstm_aggr(inputs, index, ptr, dim_size)
        #print(aggregated.shape)
        self.saved_message = aggregated.detach()
        return aggregated

    #def reset_message(self):
    #    self.saved_message = None

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')