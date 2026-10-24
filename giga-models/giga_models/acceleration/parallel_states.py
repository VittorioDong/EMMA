import torch.distributed as dist

_SEQUENCE_PARALLEL_GROUPS = dict()


def initialize_sequence_parallel_group(sp_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, 'world_size must be divisible by sequence_parallel_size'
    num_sequence_parallel_groups = world_size // sp_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp_size, (i + 1) * sp_size)
        if rank in ranks:
            group = dist.new_group(ranks)
            set_sequence_parallel_group(group)
            break


def set_sequence_parallel_group(group):
    _SEQUENCE_PARALLEL_GROUPS['sequence'] = group


def get_sequence_parallel_group():
    return _SEQUENCE_PARALLEL_GROUPS.get('sequence', None)
