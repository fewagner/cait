try: from torch.utils.data.dataloader import DataLoader
except ImportError: DataLoader = object


class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    :param sampler: The Sampler.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(DataLoader):
    """
    A child of the Pytorch data loader.
    """

    def __init__(self, *args, **kwargs):

        # CHECK IF TORCH IS INSTALLED
        if DataLoader is object: raise RuntimeError("Install 'torch>=1.8' to use this feature.")

        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)