from typing import Optional, Union, Tuple
import torch as T



class LSTMAggregation(T.nn.Module):
    r"""Performs LSTM-style aggregation in which the elements to aggregate are
    interpreted as a sequence, as described in the `"Inductive Representation
    Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. note::

        :class:`LSTMAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
        or by calling :meth:`torch_geometric.data.Data.sort`.

    .. warning::

        :class:`LSTMAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int, **kwargs):
        #print('LSTM aggr')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = T.nn.LSTM(in_channels, out_channels, batch_first=True, **kwargs)
        self.h = T.zeros(1, out_channels).cuda()
        self.c = T.zeros(1, out_channels).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def __call__(
        self,
        x: T.Tensor,
        index: Optional[T.Tensor] = None,
        ptr: Optional[T.Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> T.Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=T.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        try:
            return super().__call__(x, index=index, ptr=ptr, dim_size=dim_size,
                                    dim=dim, **kwargs)
        except (IndexError, RuntimeError) as e:
            if index is not None:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got "
                                     f"'{dim_size}' but expected "
                                     f">= '{int(index.max()) + 1}')") from e
            raise e

    def forward(
        self,
        x: T.Tensor,
        index: Optional[T.Tensor] = None,
        ptr: Optional[T.Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> T.Tensor:
        #x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
        #                           max_num_elements=max_num_elements)
        #print(x.shape)
        out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')