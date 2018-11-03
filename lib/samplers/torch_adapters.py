from torch.utils.data.sampler import Sampler


class SubmodularSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        len(self.data_source)
