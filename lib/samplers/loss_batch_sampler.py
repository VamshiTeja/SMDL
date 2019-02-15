from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch._six import int_classes as _int_classes
import time

from loss_sampler import LossSampler
from submodular import SubModSampler
from lib.utils import log
from lib.config import cfg

import numpy as np


class LossBatchSampler(Sampler):
    """
    Returns back a minibatch, which is sampled such that the SubModular Objective is maximised.

    (adapted from: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py)
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
        self.override_submodular_sampling = cfg.override_submodular_sampling
        self.submodular_sampler = LossSampler(model, data_source, self.batch_size)
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
            # r = np.random.random()
            r = 1
            print("Number of Iterations in this epoch are %d"%int(len(self.sampler)*r/self.batch_size))
            for i in range(int(len(self.sampler)*r/self.batch_size)):
                t_stamp = time.time()
                batch = self.submodular_sampler.get_subset()
                log('Fetched {0} of {1} in {2} seconds.'.format(i, len(self.sampler) // self.batch_size, time.time()-t_stamp))
                yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
