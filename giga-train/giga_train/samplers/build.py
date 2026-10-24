from giga_datasets import AspectRatioSampler, BucketBatchSampler, BucketSampler, DefaultSampler, SpecialDatasetSampler, TwoDimSampler

from ..registry import Registry, build_module

SAMPLERS = Registry()


SAMPLERS.register_module(AspectRatioSampler)
SAMPLERS.register_module(BucketBatchSampler)
SAMPLERS.register_module(BucketSampler)
SAMPLERS.register_module(DefaultSampler)
SAMPLERS.register_module(SpecialDatasetSampler)
SAMPLERS.register_module(TwoDimSampler)


def build_sampler(params_or_type, *args, **kwargs):
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)
