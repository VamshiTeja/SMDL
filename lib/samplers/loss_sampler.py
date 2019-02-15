import numpy as np
import time
import copy
import scipy
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from scipy.spatial.distance import cdist
from torch.nn.functional import normalize
from torch import Tensor
import random
import torch
import torch.nn.functional as F

from lib.config import cfg
from sampler import Sampler
from lib.utils import log


class LossSampler(Sampler):
    def __init__(self, model, dataset, batch_size, ltl_log_ep=5):
        super(LossSampler, self).__init__(model, dataset)
        self.batch_size = batch_size
        self.index_set = range(0, len(self.dataset))    # It contains the indices of each image of the set.
        self.ltl_log_ep = ltl_log_ep

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=500, shuffle=False, sampler=None,
                                              batch_sampler=None, num_workers=10)
        self.target = []
        for img in loader:
            self.target.extend(img[1])

        self.target = torch.tensor(self.target)
        self.initialize_with_activations()

    def update_activations(self, model):
        log('Updating activations with the current model...')
        self.final_activations = []
        self.penultimate_activations = []
        self.set_activations_from_model(model)
        self.initialize_with_activations()

    def initialize_with_activations(self):
        # Setup entropy
        f_acts = torch.tensor(self.final_activations)
        p_log_p = F.softmax(f_acts, dim=1) * F.log_softmax(f_acts, dim=1)
        H = -p_log_p.numpy()
        self.H = np.sum(H, axis=1)  # Compute entropy of all samples for an epoch.
        criterion = torch.nn.CrossEntropyLoss().cuda()

        self.loss = criterion(torch.tensor(self.final_activations), self.target)

    def get_subset(self, detailed_logging=False):

        subset_indices = get_subset_indices(self.index_set, self.final_activations, self.loss, self.batch_size)

        for item in subset_indices:     # Subset selection without replacement.
            self.index_set.remove(item)

        if detailed_logging:
            log('The selected {0} indices (second level): {1}'.format(len(subset_indices), subset_indices))
        return subset_indices


def get_subset_indices(index_set_input, final_activations, loss, subset_size):

    index_set = index_set_input

    subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

    subset_size = min(subset_size, len(index_set))
    for i in range(0, subset_size):
        now = time.time()
        scores = loss
        best_item_index = np.argmax(scores)
        subset_indices.append(index_set[best_item_index])
        index_set = np.delete(index_set, best_item_index, axis=0)
        # log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

    return subset_indices