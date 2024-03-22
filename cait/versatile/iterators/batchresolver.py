class BatchResolver:
    """
    Helper Class to resolve batched iterators.
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, batch):
        return [self.f(ev) for ev in batch]