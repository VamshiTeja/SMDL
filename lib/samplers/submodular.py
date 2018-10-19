from sampler import Sampler
import numpy as np


class SubmodularSampler(Sampler):
    def __init__(self, model, transforms, set, subset_size):
        super(SubmodularSampler, self).__init__(model, transforms, set, subset_size)

    def get_subset(self):
        return self._select_subset_items()

    def _select_subset_items(self, alpha_1=1, alpha_2=1, alpha_3=1, dynamic_set_size=False):
        set = self.set.copy()
        subset = []
        final_score = []

        end = len(set) if dynamic_set_size else self.subset_size
        for i in range(0, end):
            scores = []
            for item in set:
                temp_subset = list(subset)
                temp_subset.append(item)
                score = self._compute_score(temp_subset, alpha_1, alpha_2, alpha_3)
                scores.append(score)
            best_item_index = np.argmax(scores)
            best_item = set[best_item_index]

            subset.append(best_item)
            set = np.delete(set, best_item_index, axis=0)

            final_score.append(scores[best_item_index] - alpha_1*len(subset))

        if dynamic_set_size:
            subset = subset[0:np.argmax(final_score)]
        else:
            subset = subset[0:self.subset_size]

        print np.array(subset).shape
        return np.array(subset)

    def _compute_score(self, subset, alpha_1, alpha_2, alpha_3):
        """
        Compute the score for the subset.
        The score is a combination of:
            1) Diversity Score: The new point should be distant from all the elements in the class.
            2) Uncertainity Score: The point should make the model most confused.
            3) Redundancy Score: The point should be distant from all the other elements in the subset.
        :param subset:
        :param alpha_1:
        :param alpha_2:
        :param alpha_3:
        :return: The score of the subset.
        """
        score = 0
        subset = np.array(subset)
        # for item in subset:
        #     for index_in_set, img in enumerate(self.set):
        #         # if (img == item).all():
        #         #     break
        #         pass
        #
        #     # Diversity Score

        return score
