import numpy as np
import time
import torch
import torch.nn.functional as F

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

        class_mean = np.mean(self.penultimate_activations, axis=0)

        end = len(set) if dynamic_set_size else self.subset_size

        for i in range(0, end):
            now = time.time()
            scores = []

            # Compute d_score for the whole subset. Then add the d_score of just the
            # new item to compute the total d_score.
            d_score = self._compute_d_score(list(subset_indices))

            # Same logic for u_score
            u_score = self._compute_u_score(list(subset_indices))

            # Same logic for md_score
            md_score = self.compute_md_score(list(subset_indices), class_mean)

            for iter, item in enumerate(set):
                temp_subset = list(subset)
                temp_subset_indices = list(subset_indices)

                temp_subset.append(item)
                temp_subset_indices.append(index_set[iter])

                d_score += self._compute_d_score(list([index_set[iter]]))
                u_score += self._compute_u_score(list([index_set[iter]]))
                md_score += self.compute_md_score((list([index_set[iter]])), class_mean)
                r_score = self._compute_r_score(list(temp_subset_indices))

                score = d_score + u_score + r_score + md_score
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

    def _compute_d_score(self, subset_indices, alpha=1.):
        """
        Computes the Diversity Score: The new point should be distant from all the elements in the class.
        :param subset_indices:
        :param alpha:
        :return: d_score
        """
        d_score = 0
        for index in subset_indices:
            p_act = self.penultimate_activations[index]
            all_acts = self.penultimate_activations
            score = np.sum(np.linalg.norm(all_acts - p_act, axis=1))
            d_score += alpha * score

        return d_score

    def _compute_u_score(self, subset_indices, alpha=1.):
        """
        Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
        :param subset_indices:
        :param alpha:
        :return: u_score
        """
        u_score = 0
        for index in subset_indices:
            f_acts = torch.tensor(self.final_activations[index])
            p_log_p = F.softmax(f_acts, dim=0) * F.log_softmax(f_acts, dim=0)
            score = (-1.0 * p_log_p.sum()).numpy()
            u_score += alpha * score
        return u_score

    def _compute_r_score(self, subset_indices, alpha=0.2):
        """
        Computes Redundancy Score: The point should be distant from all the other elements in the subset.
        :param subset_indices:
        :param alpha:
        :return:
        """
        r_score = 0
        for index in subset_indices:
            p_act = self.penultimate_activations[index]
            if len(subset_indices) > 1:
                subset_penultimate_acts = [self.penultimate_activations[i] for i in subset_indices]
                dist = np.linalg.norm(subset_penultimate_acts - p_act, axis=1)
                score = np.min(dist[dist != 0])
                r_score += alpha * score
        return r_score

    def compute_md_score(self, subset_indices, class_mean, alpha=2.):
        """
        Computes Mean Divergence score: The new datapoint should be close to the class mean
        :param subset_indices:
        :param class_mean:
        :param alpha:
        :return:
        """
        md_score = 0
        for index in subset_indices:
            p_act = self.penultimate_activations[index]
            score = np.linalg.norm(class_mean-p_act, axis=0)
            md_score += alpha * score
        return md_score
