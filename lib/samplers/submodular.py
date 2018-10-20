import numpy as np
import multiprocessing
import time
import torch

from sampler import Sampler
from lib.utils import log

class SubmodularSampler(Sampler):
    def __init__(self, model, transforms, set, subset_size):
        super(SubmodularSampler, self).__init__(model, transforms, set, subset_size)

    def get_subset(self):
        return self._select_subset_items()

    def _select_subset_items(self, alpha_1=1, alpha_2=1, alpha_3=1, dynamic_set_size=False):
        set = self.set.copy()
        index_set = range(0, len(set))  # It contains the indices of each image of the set.
        subset = []
        subset_indices = []     # Subset of indices. Keeping track to improve computational performance.
        final_score = []

        end = len(set) if dynamic_set_size else self.subset_size

        for i in range(0, end):
            now = time.time()
            scores = []
            for iter, item in enumerate(set):
                temp_subset = list(subset)
                temp_subset_indices = list(subset_indices)

                temp_subset.append(item)
                temp_subset_indices.append(index_set[iter])

                score = self._compute_score(temp_subset_indices, alpha_1, alpha_2)
                scores.append(score)
            best_item_index = np.argmax(scores)
            best_item = set[best_item_index]

            subset.append(best_item)
            subset_indices.append(index_set[best_item_index])

            set = np.delete(set, best_item_index, axis=0)
            index_set = np.delete(index_set, best_item_index, axis=0)

            final_score.append(scores[best_item_index] - alpha_3*len(subset))
            log('Time for processing {0}/{1} exemplar is {2}'.format(i, end, time.time()-now))

        if dynamic_set_size:
            subset = subset[0:np.argmax(final_score)]
        else:
            subset = subset[0:self.subset_size]

        log(np.array(subset).shape)
        return np.array(subset)

    def _compute_score(self, subset_indices, alpha_1, alpha_2):
        """
        Compute the score for the subset.
        The score is a combination of:
            1) Diversity Score: The new point should be distant from all the elements in the class.
            2) Uncertainity Score: The point should make the model most confused.
            3) Redundancy Score: The point should be distant from all the other elements in the subset.
        :param subset:
        :param alpha_1:
        :param alpha_2:
        :return: The score of the subset.
        """
        score = 0
        for index in subset_indices:
            # Diversity Score
            p_act = self.penultimate_activations[index]
            all_acts = self.penultimate_activations
            d_score = np.sum(np.linalg.norm(all_acts - p_act, axis=1))

            # Uncertainity Score
            f_acts = torch.tensor(self.final_activations[index])
            p_log_p = self.sigmoid_module(f_acts) * self.log_sigmoid_module(f_acts)
            u_score = (-1.0 * p_log_p.sum()).numpy()

            # Redundancy Score
            r_score = 0
            if len(subset_indices) > 1:
                subset_penultimate_acts = [self.penultimate_activations[i] for i in subset_indices]
                dist = np.linalg.norm(subset_penultimate_acts - p_act, axis=1)
                r_score = np.min(dist[dist != 0])

            score += d_score + alpha_1*u_score + alpha_2*r_score
        return score
