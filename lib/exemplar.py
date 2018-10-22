import numpy as np

from lib.config import cfg
from lib.utils import log
from lib.samplers.submodular import SubmodularSampler
from lib.samplers.herding import HerdingSampler

class ExemplarManager:
    def __init__(self):
        self.exemplars = {}

    def get_exemplars(self):
        """
        Retrieve all the exemplars stores so far. Used while creating the dataset object.

        Returns:
            exemplars (dictionary)
        """
        return self.exemplars

    def add_exemplars(self, model, transforms, dataset, class_list):
        """
        Picks the exemplars from the whose set of images per class and stores it.

        Args:
            model (torch.nn.Module): The model that has been trained so far.
            transforms (torchvision.transforms): The transformationf that need to be applied to the image before forward pass.
            dataset (torch.utils.data.Dataset): The dataset object. Can retrieve the superset from this.
            class_list (np.array): The list of the new classes, for which the exemplars has to be selected.
        """
        memory_budget = cfg.dataset.memory_budget
        num_classes = len(self.exemplars) + len(class_list)
        budget_per_class = memory_budget / num_classes

        # Trim the stored exemplars
        self._trim_exemplars(budget_per_class)

        # Add new exemplars
        data, targets = dataset.get_images(class_list)
        for i, cls in enumerate(class_list):
            log('Generating exemplar set for {0}/{1} new classes.'.format(i, len(class_list)))
            imgs = data[targets == cls]
            sub_set = self._create_subset(model, transforms, imgs, budget_per_class, policy=cfg.sampling_strategy)
            self.exemplars[cls] = sub_set

        log('Current exemplar set size: ' + str(self._count_exemplars()))

    def _trim_exemplars(self, budget_per_class, policy='clear_right_end'):
        if policy == 'clear_right_end':
            for cls, imgs in self.exemplars.items():
                self.exemplars[cls] = imgs[0:budget_per_class]

    def _create_subset(self, model, transforms, set, subset_size, policy='random'):
        subset = None
        if policy == 'random':
            subset_index = np.random.choice(len(set), size=subset_size)
            subset = set[subset_index]
        elif policy == 'submodular':
            sampler = SubmodularSampler(model, transforms, set, subset_size)
            subset = sampler.get_subset()
        elif policy == 'herding':
            sampler = HerdingSampler(model, transforms, set, subset_size)
            subset = sampler.get_subset()
        else:
            raise ValueError('Recieved a wrong sampling policy: ' + policy)
        return subset

    def _count_exemplars(self, fine_logging=False):
        count = 0
        for cls, imgs in self.exemplars.items():
            if fine_logging:
                log(str(cls) + ': ' + str(len(imgs)))
            count += len(imgs)
        return count
