import torch
import torch.distributed as dist


def broadcast(input_, group):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank()
    src = rank // world_size * world_size
    input_ = input_.contiguous()
    dist.broadcast(input_, src=src, group=group)
    return input_


def _all_to_all(input_, scatter_dim, gather_dim, group):
    world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        output = _all_to_all(input_, scatter_dim, gather_dim, group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group)
        return grad_output, None, None, None


def all_to_all(input_, scatter_dim, gather_dim, group=None):
    return _AllToAll.apply(input_, scatter_dim, gather_dim, group)


def _split(input_, dim, group):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f'The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), ' f'cannot split tensor evenly'
    )
    output_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = output_list[rank].contiguous()
    return output


def _gather(input_, dim, group):
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    output_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, input_, group=group)
    output = torch.cat(output_list, dim=dim).contiguous()
    return output


class _SplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        return _split(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, ctx.group), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        return _gather(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.group), None, None


def split_forward_gather_backward(input_, dim, group=None):
    return _SplitForwardGatherBackward.apply(input_, dim, group)


def gather_forward_split_backward(input_, dim, group=None):
    return _GatherForwardSplitBackward.apply(input_, dim, group)
