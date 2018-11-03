from torch.utils.data.sampler import Sampler
from submodular import SubModSampler


class SubmodularSampler(Sampler):
    """
    Return the indices of the items selected using submodular criterion.
    The length of the returned item should be of the same length of the entire dataset.
    """
    def __init__(self, model, data_source):
        self.model = model
        self.data_source = data_source
        # self.sm_sampler = SubModSampler(self.model, self.data_source)

    def __iter__(self):
        # return iter(self.sm_sampler.get_subset())
        return iter(range(len(self.data_source)))

    def __len__(self):
        len(self.data_source)
