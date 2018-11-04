from torch.utils.data.sampler import Sampler
from submodular import SubModSampler


class SubmodularSampler(Sampler):
    """
    Return the indices of the items selected using submodular criterion.
    The length of the returned item should be of the same length of the entire dataset.
    """
    def __init__(self, model, data_source,batch_size):
        self.model = model
        self.data_source = data_source
        self.subset_size = batch_size
        self.sm_sampler = SubModSampler(self.model, self.data_source, self.subset_size)

    def __iter__(self):
        return iter(self.sm_sampler.get_subset())
        #return iter(range(len(self.data_source)))       # Just a sequential sampler for smoking.

    def __len__(self):
        len(self.data_source)

class BatchSampler (Sampler):

    def __init__(self, sampler, batch_size, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate (iter (self.sampler)):
            batch = idx
            yield batch

        if len (batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len (self.sampler) // self.batch_size