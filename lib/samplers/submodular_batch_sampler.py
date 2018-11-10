from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch._six import int_classes as _int_classes

from submodular import SubModSampler

class SubmodularBatchSampler(Sampler):
    """
    Returns back a minibatch, which is sampled such that the SubModular Objective is maximised.

    (adapted from: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py#L126)
    """

    def __init__(self, model, data_source, batch_size, sampler=None, drop_last=False):
        if sampler is None:
            sampler = RandomSampler(data_source)

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.override_submodular_sampling = False
        self.submodular_sampler = SubModSampler(model, data_source, self.batch_size)
        # TODO: Handle Replacement Strategy

    def __iter__(self):
        batch = []
        if self.override_submodular_sampling:
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        else:
            for i in range(len(self.sampler) // self.batch_size):
                batch = self.submodular_sampler.get_subset()
                yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size