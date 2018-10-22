import numpy as np
import time
import torch
import torch.nn.functional as F

from sampler import Sampler
from lib.utils import log


class HerdingSampler(Sampler):
    def __init__(self, model, transforms, set, subset_size):
        super(HerdingSampler, self).__init__(model, transforms, set, subset_size)

    def get_subset(self):
        return self._select_subset_items()

    def _select_subset_items(self):
        set = self.set.copy()
        index_set = range(0, len(set))  # It contains the indices of each image of the set.
        subset = []
        subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

        class_mean = np.mean(self.penultimate_activations, axis=0)

        for i in range(0, self.subset_size):
            now = time.time()
            scores = []
            for iter, item in enumerate(set):
                temp_subset = list(subset)
                temp_subset_indices = list(subset_indices)

                temp_subset.append(item)
                temp_subset_indices.append(index_set[iter])

                score = self._compute_score(class_mean, temp_subset_indices)
                scores.append(score)
            best_item_index = np.argmin(scores)
            best_item = set[best_item_index]

            subset.append(best_item)
            subset_indices.append(index_set[best_item_index])

            set = np.delete(set, best_item_index, axis=0)
            index_set = np.delete(index_set, best_item_index, axis=0)

            log('Time for processing {0}/{1} exemplar is {2}'.format(i, self.subset_size, time.time()-now))

        log(np.array(subset).shape)
        return np.array(subset)

    def _compute_score(self, class_mean, subset_indices):
        """
        Compute the score for the subset.
        :param subset_indices:
        :return: The score of the subset.
        """

        subset_penultimate_acts = [self.penultimate_activations[i] for i in subset_indices]
        avg = np.average(subset_penultimate_acts, axis=0)
        score = np.linalg.norm(class_mean - avg, axis=0)

        return score
